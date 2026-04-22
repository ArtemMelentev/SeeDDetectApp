package com.seeddetect.app

import android.app.Activity
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.Toast

class ShareReceiverActivity : Activity() {
    private val importer: SharedContentImporter by lazy { SharedContentImporter(applicationContext) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        handleIncomingIntent(intent)
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        setIntent(intent)
        handleIncomingIntent(intent)
    }

    private fun handleIncomingIntent(intent: Intent?) {
        if (intent == null) {
            finishWithError("Не удалось получить данные для импорта")
            return
        }

        val action = intent.action
        val intentMime = intent.type
        Log.i(
            TAG,
            "[Share][IMP:6][handleIncomingIntent][INTENT][Flow] action=$action intentMime=$intentMime [INFO]",
        )

        when (action) {
            Intent.ACTION_SEND -> {
                val uri = intent.readSingleUriExtra()
                if (uri == null) {
                    finishWithError("Не удалось получить URI файла")
                    return
                }

                processBatch(action, listOf(uri), intentMime)
            }

            Intent.ACTION_SEND_MULTIPLE -> {
                val uris = intent.readMultipleUriExtra()
                if (uris.isEmpty()) {
                    finishWithError("Список переданных файлов пуст")
                    return
                }

                processBatch(action, uris, intentMime)
            }

            else -> {
                finishWithError("Неподдерживаемый тип отправки файла")
            }
        }
    }

    private fun processBatch(action: String, uris: List<Uri>, intentMime: String?) {
        val imported = mutableListOf<SharedContentImporter.ImportedContent>()
        val errors = mutableListOf<SharedContentImporter.ImportError>()

        uris.forEachIndexed { index, uri ->
            Log.i(
                TAG,
                "[Share][IMP:6][processBatch][ITEM][Flow] index=$index uri=${toSafeLogUri(uri)} [INFO]",
            )

            val result = importer.importSharedUri(uri, intentMime)
            if (result.success && result.payload != null) {
                imported.add(result.payload)
            } else if (result.error != null) {
                errors.add(result.error)
            } else {
                errors.add(
                    SharedContentImporter.ImportError(
                        code = "unknown_import_error",
                        message = "Неизвестная ошибка импорта",
                    ),
                )
            }
        }

        if (imported.isEmpty()) {
            val errorMessage = errors.firstOrNull()?.message ?: "Не удалось импортировать переданные файлы"
            finishWithError(errorMessage)
            return
        }

        val summaryMessage = buildSummaryMessage(total = uris.size, imported = imported.size, failed = errors.size)
        val primary = imported.first()
        launchMainWithImportedContent(
            action = action,
            primary = primary,
            totalCount = uris.size,
            importedCount = imported.size,
            failedCount = errors.size,
            summaryMessage = summaryMessage,
        )
    }

    private fun launchMainWithImportedContent(
        action: String,
        primary: SharedContentImporter.ImportedContent,
        totalCount: Int,
        importedCount: Int,
        failedCount: Int,
        summaryMessage: String,
    ) {
        Log.i(
            TAG,
            "[Share][IMP:7][launchMainWithImportedContent][NAVIGATION][Flow] action=$action imported=$importedCount failed=$failedCount total=$totalCount [SUCCESS]",
        )

        val launchIntent = Intent(this, MainActivity::class.java).apply {
            addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP or Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_NEW_TASK)
            putExtra(ShareIntentContract.EXTRA_SHARED_INPUT_URI, primary.localUri)
            putExtra(ShareIntentContract.EXTRA_SHARED_MIME_TYPE, primary.mimeType)
            putExtra(ShareIntentContract.EXTRA_SHARED_DISPLAY_NAME, primary.displayName)
            putExtra(ShareIntentContract.EXTRA_SHARED_SOURCE_ACTION, action)
            putExtra(ShareIntentContract.EXTRA_SHARED_TOTAL_COUNT, totalCount)
            putExtra(ShareIntentContract.EXTRA_SHARED_IMPORTED_COUNT, importedCount)
            putExtra(ShareIntentContract.EXTRA_SHARED_FAILED_COUNT, failedCount)
            putExtra(ShareIntentContract.EXTRA_SHARED_SUMMARY_MESSAGE, summaryMessage)
        }

        startActivity(launchIntent)
        finish()
    }

    private fun finishWithError(message: String) {
        Log.e(TAG, "[Share][IMP:9][finishWithError][TERMINATE][Exception] message=$message [FAIL]")
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        finish()
    }

    private fun buildSummaryMessage(total: Int, imported: Int, failed: Int): String {
        if (total <= 1) {
            return "Файл импортирован. Открываем в приложении."
        }

        return if (failed > 0) {
            "Импортировано $imported из $total файлов. Открыт первый поддерживаемый файл."
        } else {
            "Импортировано $imported файлов. Открыт первый файл."
        }
    }

    @Suppress("DEPRECATION")
    private fun Intent.readSingleUriExtra(): Uri? {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        } else {
            getParcelableExtra(Intent.EXTRA_STREAM)
        }
    }

    @Suppress("DEPRECATION")
    private fun Intent.readMultipleUriExtra(): List<Uri> {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java) ?: emptyList()
        } else {
            getParcelableArrayListExtra<Uri>(Intent.EXTRA_STREAM) ?: emptyList()
        }
    }

    private fun toSafeLogUri(uri: Uri): String {
        val scheme = uri.scheme ?: "unknown"
        val authority = uri.authority ?: "-"
        val tail = uri.lastPathSegment?.takeLast(32) ?: "-"
        return "$scheme://$authority/$tail"
    }

    companion object {
        private const val TAG = "SeedDetectShareReceiver"
    }
}
