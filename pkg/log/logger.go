package log

import (
	"fmt"
	"os"
	"strings"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/cockroachdb/errors"
)

// GlobalLogger はプロジェクト全体で使用するzerologインスタンスです。
var GlobalLogger zerolog.Logger

// SetupLogger はグローバルロガーを設定します。
// CloudLogging形式への変換とcockroachdb/errorsのスタックトレース対応を含みます。
func SetupLogger(loglevel string) {
	level := ToLogLevel(loglevel)
	
	// CloudLogging形式のフィールドマッピングを設定
	zerolog.LevelFieldName = "severity"
	zerolog.MessageFieldName = "message"
	zerolog.TimestampFieldName = "timestamp"
	
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	zerolog.SetGlobalLevel(level)

	// カスタマイズされたコンソールライター（CloudLogging互換）
	output := zerolog.ConsoleWriter{
		Out:        os.Stdout,
		NoColor:    true, // JSON形式で出力
		TimeFormat: zerolog.TimeFormatUnix,
	}
	
	GlobalLogger = zerolog.New(output).With().
		Timestamp().
		Caller().
		Logger()

	// グローバルロガーとして設定
	log.Logger = GlobalLogger
}

// ToLogLevel はログレベル文字列をzerologのレベルに変換します。
func ToLogLevel(level string) zerolog.Level {
	switch strings.ToLower(level) {
	case "info":
		return zerolog.InfoLevel
	case "debug":
		return zerolog.DebugLevel
	case "warn", "warning":
		return zerolog.WarnLevel
	case "error":
		return zerolog.ErrorLevel
	case "fatal":
		return zerolog.FatalLevel
	case "panic":
		return zerolog.PanicLevel
	case "disabled":
		return zerolog.Disabled
	default:
		panic(fmt.Sprintf("invalid log level: %s", level))
	}
}

const (
	ErrAttrKey        = "error"
	StacktraceAttrKey = "stacktrace"
)

// GetLogger は設定済みのグローバルロガーを返します。
func GetLogger() zerolog.Logger {
	return GlobalLogger
}

// LogError はcockroachdb/errorsと統合されたエラーログを出力します。
// scikit-learnスタイルの詳細なエラー情報を構造化ログとして記録します。
func LogError(err error, msg string) {
	event := GlobalLogger.Error().Err(err)
	
	// zerolog.LogObjectMarshalerインターフェースを実装している場合は構造化データを追加
	if marshaler, ok := err.(zerolog.LogObjectMarshaler); ok {
		event = event.Object("details", marshaler)
	}
	
	// cockroachdb/errorsのスタックトレースを抽出
	stacktrace := extractStacktrace(err)
	if stacktrace != "" {
		event = event.Str(StacktraceAttrKey, stacktrace)
	}
	
	event.Msg(msg)
}

// LogWarningWithDetails は構造化された警告ログを出力します。
func LogWarningWithDetails(warning error) {
	event := GlobalLogger.Warn()
	
	// zerolog.LogObjectMarshalerインターフェースを実装している場合は構造化データを追加
	if marshaler, ok := warning.(zerolog.LogObjectMarshaler); ok {
		event = event.Object("warning_details", marshaler)
	}
	
	event.Msg(warning.Error())
}

// LogWarn は警告ログを出力します。
func LogWarn(msg string) {
	GlobalLogger.Warn().Msg(msg)
}

// LogInfo は情報ログを出力します。  
func LogInfo(msg string) {
	GlobalLogger.Info().Msg(msg)
}

// LogDebug はデバッグログを出力します。
func LogDebug(msg string) {
	GlobalLogger.Debug().Msg(msg)
}

// extractStacktrace はcockroachdb/errorsからスタックトレースを抽出します。
func extractStacktrace(err error) string {
	safeDetails := errors.GetSafeDetails(err).SafeDetails
	if len(safeDetails) > 0 {
		return safeDetails[0]
	}
	return ""
}