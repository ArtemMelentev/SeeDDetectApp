package com.seeddetect.app

import android.content.Context
import android.database.Cursor
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import android.webkit.MimeTypeMap
import java.io.File
import java.io.FileNotFoundException
import java.io.IOException
import java.util.Locale
import java.util.UUID

class SharedContentImporter(private val context: Context) {
    fun importSharedUri(uri: Uri, fallbackMimeType: String?): ImportResult {
        val safeUri = toSafeLogUri(uri)
        Log.i(
            TAG,
            "[Share][IMP:6][importSharedUri][START][Flow] uri=$safeUri fallbackMime=$fallbackMimeType [INFO]",
        )

        val metadata = readMetadata(uri, fallbackMimeType)
        val normalizedMime = metadata.mimeType?.lowercase(Locale.US)
        Log.i(
            TAG,
            "[Share][IMP:6][importSharedUri][METADATA][Flow] mime=$normalizedMime displayName=${metadata.displayName} size=${metadata.sizeBytes} [INFO]",
        )

        if (!isSupportedMime(normalizedMime)) {
            return ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "unsupported_mime",
                    message = "Формат файла не поддерживается",
                    details = "mime=$normalizedMime uri=$safeUri",
                ),
            )
        }

        val targetDir = File(context.cacheDir, IMPORT_DIRECTORY_NAME)
        if (!targetDir.exists() && !targetDir.mkdirs()) {
            return ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "target_dir_unavailable",
                    message = "Не удалось подготовить внутреннее хранилище",
                    details = targetDir.absolutePath,
                ),
            )
        }

        val targetFile = File(targetDir, buildSafeTargetFileName(metadata.displayName, normalizedMime))

        return try {
            val copiedBytes = context.contentResolver.openInputStream(uri)?.use { input ->
                targetFile.outputStream().use { output ->
                    input.copyTo(output)
                }
            } ?: return ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "stream_unavailable",
                    message = "Не удалось открыть входящий файл",
                    details = "uri=$safeUri",
                ),
            )

            if (copiedBytes <= 0L) {
                targetFile.delete()
                return ImportResult(
                    success = false,
                    payload = null,
                    error = ImportError(
                        code = "empty_stream",
                        message = "Получен пустой файл",
                        details = "uri=$safeUri",
                    ),
                )
            }

            val payload = ImportedContent(
                sourceUri = uri.toString(),
                localPath = targetFile.absolutePath,
                localUri = Uri.fromFile(targetFile).toString(),
                mimeType = normalizedMime ?: "application/octet-stream",
                displayName = metadata.displayName ?: targetFile.name,
                sizeBytes = copiedBytes,
            )

            Log.i(
                TAG,
                "[Share][IMP:7][importSharedUri][COPY_RESULT][I/O] uri=$safeUri localPath=${targetFile.absolutePath} bytes=$copiedBytes [SUCCESS]",
            )

            ImportResult(success = true, payload = payload, error = null)
        } catch (exc: SecurityException) {
            Log.e(
                TAG,
                "[Share][IMP:10][importSharedUri][COPY_ERROR][Exception] SecurityException uri=$safeUri [FAIL]",
                exc,
            )
            ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "security_error",
                    message = "Нет доступа к переданному файлу",
                    details = exc.message,
                ),
            )
        } catch (exc: FileNotFoundException) {
            Log.e(
                TAG,
                "[Share][IMP:10][importSharedUri][COPY_ERROR][Exception] FileNotFoundException uri=$safeUri [FAIL]",
                exc,
            )
            ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "file_not_found",
                    message = "Файл не найден или недоступен",
                    details = exc.message,
                ),
            )
        } catch (exc: IOException) {
            Log.e(
                TAG,
                "[Share][IMP:10][importSharedUri][COPY_ERROR][Exception] IOException uri=$safeUri [FAIL]",
                exc,
            )
            ImportResult(
                success = false,
                payload = null,
                error = ImportError(
                    code = "io_error",
                    message = "Ошибка чтения входящего файла",
                    details = exc.message,
                ),
            )
        }
    }

    private fun readMetadata(uri: Uri, fallbackMimeType: String?): SharedMetadata {
        var displayName: String? = null
        var sizeBytes: Long? = null

        runCatching {
            context.contentResolver.query(
                uri,
                arrayOf(OpenableColumns.DISPLAY_NAME, OpenableColumns.SIZE),
                null,
                null,
                null,
            )
        }.getOrNull()?.use { cursor ->
            if (cursor.moveToFirst()) {
                displayName = cursor.readStringSafe(OpenableColumns.DISPLAY_NAME)
                sizeBytes = cursor.readLongSafe(OpenableColumns.SIZE)
            }
        }

        val mimeFromResolver = runCatching {
            context.contentResolver.getType(uri)
        }.getOrNull()?.lowercase(Locale.US)

        val normalizedFallback = fallbackMimeType?.lowercase(Locale.US)
        val mimeType = when {
            !mimeFromResolver.isNullOrBlank() -> mimeFromResolver
            !normalizedFallback.isNullOrBlank() && normalizedFallback != "*/*" -> normalizedFallback
            else -> guessMimeTypeFromName(displayName ?: uri.lastPathSegment)
        }

        return SharedMetadata(
            displayName = displayName,
            sizeBytes = sizeBytes,
            mimeType = mimeType,
        )
    }

    private fun guessMimeTypeFromName(rawName: String?): String? {
        if (rawName.isNullOrBlank()) {
            return null
        }

        val ext = rawName.substringAfterLast('.', missingDelimiterValue = "")
            .lowercase(Locale.US)
            .trim()
        if (ext.isBlank()) {
            return null
        }

        return MimeTypeMap.getSingleton().getMimeTypeFromExtension(ext)
    }

    private fun isSupportedMime(mimeType: String?): Boolean {
        if (mimeType.isNullOrBlank()) {
            return false
        }

        return mimeType == MIME_PDF || mimeType.startsWith(MIME_IMAGE_PREFIX)
    }

    private fun buildSafeTargetFileName(displayName: String?, mimeType: String?): String {
        val extension = resolveExtension(displayName, mimeType)
        val rawBase = displayName
            ?.substringAfterLast('/')
            ?.substringAfterLast('\\')
            ?.substringBeforeLast('.')
            ?.trim()
            .orEmpty()

        val safeBase = rawBase
            .replace(Regex("[^A-Za-z0-9._-]"), "_")
            .trim('_', '.', '-')
            .take(48)
            .ifBlank { "shared_item" }

        return "${safeBase}_${UUID.randomUUID()}.$extension"
    }

    private fun resolveExtension(displayName: String?, mimeType: String?): String {
        val extFromName = displayName
            ?.substringAfterLast('.', missingDelimiterValue = "")
            ?.lowercase(Locale.US)
            ?.takeIf { it.matches(Regex("[a-z0-9]{1,8}")) }
        if (!extFromName.isNullOrBlank()) {
            return extFromName
        }

        val extFromMime = mimeType
            ?.let { MimeTypeMap.getSingleton().getExtensionFromMimeType(it) }
            ?.lowercase(Locale.US)
            ?.takeIf { it.matches(Regex("[a-z0-9]{1,8}")) }
        if (!extFromMime.isNullOrBlank()) {
            return extFromMime
        }

        return if (mimeType == MIME_PDF) "pdf" else "jpg"
    }

    private fun Cursor.readStringSafe(columnName: String): String? {
        val idx = getColumnIndex(columnName)
        if (idx < 0 || isNull(idx)) {
            return null
        }

        return getString(idx)
    }

    private fun Cursor.readLongSafe(columnName: String): Long? {
        val idx = getColumnIndex(columnName)
        if (idx < 0 || isNull(idx)) {
            return null
        }

        return getLong(idx)
    }

    private fun toSafeLogUri(uri: Uri): String {
        val scheme = uri.scheme ?: "unknown"
        val authority = uri.authority ?: "-"
        val tail = uri.lastPathSegment?.takeLast(32) ?: "-"
        return "$scheme://$authority/$tail"
    }

    data class SharedMetadata(
        val displayName: String?,
        val sizeBytes: Long?,
        val mimeType: String?,
    )

    data class ImportedContent(
        val sourceUri: String,
        val localPath: String,
        val localUri: String,
        val mimeType: String,
        val displayName: String,
        val sizeBytes: Long,
    )

    data class ImportError(
        val code: String,
        val message: String,
        val details: String? = null,
    )

    data class ImportResult(
        val success: Boolean,
        val payload: ImportedContent?,
        val error: ImportError?,
    )

    companion object {
        private const val TAG = "SeedDetectShareImport"
        private const val MIME_PDF = "application/pdf"
        private const val MIME_IMAGE_PREFIX = "image/"
        private const val IMPORT_DIRECTORY_NAME = "shared_imports"
    }
}
