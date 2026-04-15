package com.seeddetect.app

import android.net.Uri
import android.webkit.MimeTypeMap
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
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

        val inputFile = copyInputToInternal(inputUri, runDir)

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
        val fromPath = uri.lastPathSegment ?: uri.path ?: inputUri
        val rawExt = MimeTypeMap.getFileExtensionFromUrl(fromPath)
        if (!rawExt.isNullOrBlank()) {
            return ".${rawExt.lowercase(Locale.US)}"
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

    companion object {
        private const val CHANNEL = "seed_detect/analyzer"
    }
}
