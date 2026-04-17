package com.seeddetect.app

import android.graphics.Bitmap
import android.net.Uri
import android.util.Log
import android.webkit.MimeTypeMap
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDResources
import com.tom_roush.pdfbox.pdmodel.graphics.form.PDFormXObject
import com.tom_roush.pdfbox.pdmodel.graphics.image.PDImageXObject
import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.io.InputStream
import java.util.Locale
import java.util.UUID

class MainActivity : FlutterActivity() {
    private val ioScope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    @Volatile
    private var pdfBoxInitialized = false

    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)

        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, CHANNEL)
            .setMethodCallHandler { call, result ->
                when (call.method) {
                    "analyzeImage" -> handleAnalyze(call, result)
                    else -> result.notImplemented()
                }
            }
    }

    override fun onDestroy() {
        ioScope.cancel()
        super.onDestroy()
    }

    private fun handleAnalyze(call: MethodCall, result: MethodChannel.Result) {
        val inputUri = call.argument<String>("inputUri")
        if (inputUri.isNullOrBlank()) {
            result.success(errorPayload("invalid_args", "Missing inputUri"))
            return
        }

        ioScope.launch {
            val payload = try {
                runAnalysis(inputUri)
            } catch (exc: PdfProcessingException) {
                errorPayload(exc.code, exc.messageForUser, exc.details)
            } catch (exc: Exception) {
                errorPayload("native_error", "Native bridge failed", exc.message)
            }

            withContext(Dispatchers.Main) {
                result.success(payload)
            }
        }
    }

    private fun runAnalysis(inputUri: String): Map<String, Any?> {
        val runRoot = File(cacheDir, "seed_runs")
        if (!runRoot.exists()) {
            runRoot.mkdirs()
        }
        cleanupOldRuns(runRoot, 24L * 60L * 60L * 1000L)

        val runDir = File(runRoot, "run_${System.currentTimeMillis()}_${UUID.randomUUID()}")
        if (!runDir.mkdirs()) {
            throw IOException("Cannot create run directory: ${runDir.absolutePath}")
        }

        val inputFile = prepareInputForAnalysis(inputUri, runDir)

        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(applicationContext))
        }

        val py = Python.getInstance()
        val bridge = py.getModule("android_bridge")
        val jsonPayload = bridge.callAttr(
            "run_analysis_json",
            inputFile.absolutePath,
            runDir.absolutePath,
        ).toString()

        val payload = jsonObjectToMap(JSONObject(jsonPayload)).toMutableMap()
        payload["run_dir"] = runDir.absolutePath
        return payload
    }

    private fun prepareInputForAnalysis(inputUri: String, runDir: File): File {
        val isPdf = isPdfInput(inputUri)
        Log.i(TAG, "Input classification. uri=$inputUri isPdf=$isPdf guessedExt=${guessExtension(inputUri)}")

        return if (isPdf) {
            extractSingleImageFromPdf(inputUri, runDir)
        } else {
            copyInputToInternal(inputUri, runDir)
        }
    }

    private fun copyInputToInternal(inputUri: String, runDir: File): File {
        val ext = guessExtension(inputUri)
        val targetFile = File(runDir, "input$ext")

        resolveInputStream(inputUri).use { input ->
            targetFile.outputStream().use { output ->
                input.copyTo(output)
            }
        }

        return targetFile
    }

    private fun extractSingleImageFromPdf(inputUri: String, runDir: File): File {
        ensurePdfBoxInitialized()
        Log.i(TAG, "Starting PDF image extraction. uri=$inputUri")

        try {
            resolveInputStream(inputUri).use { input ->
                PDDocument.load(input).use { document ->
                    val images = mutableListOf<PDImageXObject>()
                    val stats = PdfCollectStats()
                    val pageCount = document.numberOfPages
                    Log.i(TAG, "PDF loaded. pages=$pageCount")

                    for ((pageIndex, page) in document.pages.withIndex()) {
                        collectPdfImages(
                            resources = page.resources,
                            target = images,
                            pageIndex = pageIndex,
                            depth = 0,
                            visited = mutableSetOf(),
                            stats = stats,
                        )
                    }

                    Log.i(TAG, "PDF scan completed. ${stats.asLogString()}, found_images=${images.size}")

                    if (images.isEmpty()) {
                        val details = "${stats.asLogString()}, pages=$pageCount"
                        Log.w(TAG, "No extractable PDF images found. $details")
                        throw PdfProcessingException(
                            code = "pdf_image_not_found",
                            messageForUser = "PDF does not contain an embedded image",
                            details = details,
                        )
                    }

                    if (images.size > 1) {
                        val imageDetails = images
                            .take(5)
                            .joinToString(separator = " | ") { buildPdfImageMeta(it) }
                        val details = "found_images=${images.size}; ${stats.asLogString()}; sample_images=$imageDetails"
                        Log.w(TAG, "Multiple PDF images found. $details")
                        throw PdfProcessingException(
                            code = "pdf_multiple_images",
                            messageForUser = "PDF contains multiple images",
                            details = details,
                        )
                    }

                    Log.i(TAG, "Single PDF image selected: ${buildPdfImageMeta(images.first())}")
                    return writeExtractedPdfImage(images.first(), runDir)
                }
            }
        } catch (exc: PdfProcessingException) {
            throw exc
        } catch (exc: Exception) {
            Log.e(TAG, "PDF extraction failed. uri=$inputUri", exc)
            throw PdfProcessingException(
                code = "pdf_extract_failed",
                messageForUser = "Failed to extract image from PDF",
                details = "${exc.javaClass.simpleName}: ${exc.message}",
                cause = exc,
            )
        }
    }

    private fun collectPdfImages(
        resources: PDResources?,
        target: MutableList<PDImageXObject>,
        pageIndex: Int,
        depth: Int,
        visited: MutableSet<PDResources> = mutableSetOf(),
        stats: PdfCollectStats,
    ) {
        if (resources == null || visited.contains(resources)) {
            return
        }

        visited.add(resources)
        stats.resourcesVisited += 1
        if (depth > stats.maxDepth) {
            stats.maxDepth = depth
        }

        for (name in resources.xObjectNames) {
            val xObject = resources.getXObject(name)
            when (xObject) {
                is PDImageXObject -> {
                    target.add(xObject)
                    stats.imageObjects += 1
                    Log.d(
                        TAG,
                        "PDF xobject image. page=$pageIndex depth=$depth name=$name ${buildPdfImageMeta(xObject)}",
                    )
                }

                is PDFormXObject -> {
                    stats.formObjects += 1
                    Log.d(TAG, "PDF xobject form. page=$pageIndex depth=$depth name=$name")
                    collectPdfImages(
                        resources = xObject.resources,
                        target = target,
                        pageIndex = pageIndex,
                        depth = depth + 1,
                        visited = visited,
                        stats = stats,
                    )
                }

                else -> {
                    stats.otherObjects += 1
                    Log.d(
                        TAG,
                        "PDF xobject other. page=$pageIndex depth=$depth name=$name type=${xObject.javaClass.simpleName}",
                    )
                }
            }
        }
    }

    private fun writeExtractedPdfImage(image: PDImageXObject, runDir: File): File {
        val sourceSuffix = image.suffix?.lowercase(Locale.US)
        val useJpeg = sourceSuffix == "jpg" || sourceSuffix == "jpeg"
        val targetExt = if (useJpeg) ".jpg" else ".png"
        val targetFile = File(runDir, "input_extracted$targetExt")
        Log.i(TAG, "Encoding extracted PDF image. ${buildPdfImageMeta(image)}, target=$targetExt")

        val bitmap = try {
            image.image
        } catch (exc: Exception) {
            throw IOException("PDF image decode failed: ${buildPdfImageMeta(image)}", exc)
        } ?: throw IOException("PDF image cannot be decoded: ${buildPdfImageMeta(image)}")

        try {
            targetFile.outputStream().use { output ->
                val encoded = bitmap.compress(
                    if (useJpeg) Bitmap.CompressFormat.JPEG else Bitmap.CompressFormat.PNG,
                    if (useJpeg) 95 else 100,
                    output,
                )
                if (!encoded) {
                    throw IOException("Failed to encode extracted PDF image")
                }
            }
        } finally {
            bitmap.recycle()
        }

        return targetFile
    }

    private fun buildPdfImageMeta(image: PDImageXObject): String {
        val suffix = image.suffix ?: "unknown"
        return "suffix=$suffix size=${image.width}x${image.height} bits=${image.bitsPerComponent} stencil=${image.isStencil}"
    }

    private fun ensurePdfBoxInitialized() {
        if (pdfBoxInitialized) {
            return
        }

        synchronized(this) {
            if (!pdfBoxInitialized) {
                PDFBoxResourceLoader.init(applicationContext)
                pdfBoxInitialized = true
            }
        }
    }

    private fun resolveInputStream(inputUri: String): InputStream {
        val uri = Uri.parse(inputUri)
        return when (uri.scheme?.lowercase(Locale.US)) {
            "content", "android.resource" -> {
                contentResolver.openInputStream(uri)
                    ?: throw IOException("Cannot open content URI: $inputUri")
            }

            "file" -> {
                val path = uri.path ?: throw IOException("Invalid file URI: $inputUri")
                FileInputStream(path)
            }

            null -> {
                val file = File(inputUri)
                if (!file.exists()) {
                    throw IOException("Input file not found: $inputUri")
                }
                FileInputStream(file)
            }

            else -> {
                contentResolver.openInputStream(uri)
                    ?: throw IOException("Cannot open URI with scheme ${uri.scheme}")
            }
        }
    }

    private fun guessExtension(inputUri: String): String {
        val uri = Uri.parse(inputUri)

        val pathCandidates = listOfNotNull(uri.lastPathSegment, uri.path, inputUri)
        for (candidate in pathCandidates) {
            val extFromPath = extractExtensionFromPath(candidate)
            if (!extFromPath.isNullOrBlank()) {
                return ".$extFromPath"
            }

            val decodedCandidate = Uri.decode(candidate)
            if (decodedCandidate != candidate) {
                val extFromDecoded = extractExtensionFromPath(decodedCandidate)
                if (!extFromDecoded.isNullOrBlank()) {
                    return ".$extFromDecoded"
                }
            }
        }

        val mime = try {
            contentResolver.getType(uri)
        } catch (_: Exception) {
            null
        }

        if (!mime.isNullOrBlank()) {
            val ext = MimeTypeMap.getSingleton().getExtensionFromMimeType(mime)
            if (!ext.isNullOrBlank()) {
                return ".${ext.lowercase(Locale.US)}"
            }
        }

        return ".jpg"
    }

    private fun isPdfInput(inputUri: String): Boolean {
        val uri = Uri.parse(inputUri)
        val mime = try {
            contentResolver.getType(uri)
        } catch (_: Exception) {
            null
        }

        if (mime.equals("application/pdf", ignoreCase = true)) {
            return true
        }

        return extractExtensionFromPath(uri.lastPathSegment) == "pdf"
            || extractExtensionFromPath(uri.path) == "pdf"
            || extractExtensionFromPath(inputUri) == "pdf"
            || extractExtensionFromPath(Uri.decode(inputUri)) == "pdf"
    }

    private fun extractExtensionFromPath(pathValue: String?): String? {
        if (pathValue.isNullOrBlank()) {
            return null
        }

        val withoutQuery = pathValue.substringBefore('?').substringBefore('#')
        val fileName = withoutQuery.substringAfterLast('/').substringAfterLast('\\')
        val dotIndex = fileName.lastIndexOf('.')
        if (dotIndex <= 0 || dotIndex == fileName.lastIndex) {
            return null
        }

        val ext = fileName.substring(dotIndex + 1).lowercase(Locale.US)
        if (!ext.all { it in 'a'..'z' || it in '0'..'9' }) {
            return null
        }

        return ext
    }

    private fun cleanupOldRuns(root: File, ttlMs: Long) {
        val cutoff = System.currentTimeMillis() - ttlMs
        root.listFiles()?.forEach { file ->
            if (file.isDirectory && file.lastModified() < cutoff) {
                file.deleteRecursively()
            }
        }
    }

    private fun jsonObjectToMap(jsonObject: JSONObject): Map<String, Any?> {
        val map = mutableMapOf<String, Any?>()
        val iterator = jsonObject.keys()
        while (iterator.hasNext()) {
            val key = iterator.next()
            map[key] = jsonValue(jsonObject.get(key))
        }
        return map
    }

    private fun jsonArrayToList(jsonArray: JSONArray): List<Any?> {
        val list = mutableListOf<Any?>()
        for (i in 0 until jsonArray.length()) {
            list.add(jsonValue(jsonArray.get(i)))
        }
        return list
    }

    private fun jsonValue(value: Any?): Any? {
        return when (value) {
            is JSONObject -> jsonObjectToMap(value)
            is JSONArray -> jsonArrayToList(value)
            JSONObject.NULL -> null
            else -> value
        }
    }

    private fun errorPayload(code: String, message: String, details: String? = null): Map<String, Any?> {
        val error = mutableMapOf<String, Any?>(
            "code" to code,
            "message" to message,
        )
        if (!details.isNullOrBlank()) {
            error["details"] = details
        }
        return mapOf(
            "ok" to false,
            "error" to error,
        )
    }

    private class PdfProcessingException(
        val code: String,
        val messageForUser: String,
        val details: String? = null,
        cause: Throwable? = null,
    ) : Exception(messageForUser, cause)

    private data class PdfCollectStats(
        var resourcesVisited: Int = 0,
        var imageObjects: Int = 0,
        var formObjects: Int = 0,
        var otherObjects: Int = 0,
        var maxDepth: Int = 0,
    ) {
        fun asLogString(): String {
            return "resources=$resourcesVisited,image_xobj=$imageObjects,form_xobj=$formObjects,other_xobj=$otherObjects,max_depth=$maxDepth"
        }
    }

    companion object {
        private const val CHANNEL = "seed_detect/analyzer"
        private const val TAG = "SeedDetectPdf"
    }
}
