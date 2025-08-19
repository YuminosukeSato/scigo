package log

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/cockroachdb/errors"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// GlobalLogger はプロジェクト全体で使用するzerologインスタンスです。
var GlobalLogger zerolog.Logger

// globalProvider is the default logger provider instance.
var globalProvider LoggerProvider

// SetupLogger はグローバルロガーを設定します。
func SetupLogger(loglevel string) {
	level := ToLogLevel(loglevel)

	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	zerolog.SetGlobalLevel(level)

	// JSON形式で標準出力に出力
	GlobalLogger = zerolog.New(os.Stdout).With().
		Timestamp().
		Caller().
		Logger()

	// グローバルロガーとして設定
	log.Logger = GlobalLogger

	// Initialize the global provider
	globalProvider = NewZerologProvider(level)
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

// Legacy constants for backward compatibility
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

// zerologLogger implements the Logger interface using zerolog as the backend.
type zerologLogger struct {
	logger zerolog.Logger
}

// Debug implements Logger.Debug.
func (z *zerologLogger) Debug(msg string, fields ...any) {
	event := z.logger.Debug()
	z.addFields(event, fields...)
	event.Msg(msg)
}

// Info implements Logger.Info.
func (z *zerologLogger) Info(msg string, fields ...any) {
	event := z.logger.Info()
	z.addFields(event, fields...)
	event.Msg(msg)
}

// Warn implements Logger.Warn.
func (z *zerologLogger) Warn(msg string, fields ...any) {
	event := z.logger.Warn()
	z.addFields(event, fields...)
	event.Msg(msg)
}

// Error implements Logger.Error.
func (z *zerologLogger) Error(msg string, fields ...any) {
	event := z.logger.Error()

	// Check if the first field is an error for special handling
	if len(fields) > 0 {
		if err, ok := fields[0].(error); ok {
			event = event.Err(err)

			// Add stack trace if available
			stacktrace := extractStacktrace(err)
			if stacktrace != "" {
				event = event.Str(StacktraceAttrKey, stacktrace)
			}

			// Process remaining fields
			z.addFields(event, fields[1:]...)
		} else {
			z.addFields(event, fields...)
		}
	}

	event.Msg(msg)
}

// With implements Logger.With.
func (z *zerologLogger) With(fields ...any) Logger {
	logger := z.logger.With()
	z.addFieldsToContext(logger, fields...)
	return &zerologLogger{logger: logger.Logger()}
}

// Enabled implements Logger.Enabled.
func (z *zerologLogger) Enabled(ctx context.Context, level Level) bool {
	zerologLevel := z.convertLevel(level)
	return z.logger.GetLevel() <= zerologLevel
}

// addFields adds key-value pairs to a zerolog event.
func (z *zerologLogger) addFields(event *zerolog.Event, fields ...any) {
	for i := 0; i < len(fields)-1; i += 2 {
		key := fmt.Sprintf("%v", fields[i])
		value := fields[i+1]

		switch v := value.(type) {
		case string:
			event.Str(key, v)
		case int:
			event.Int(key, v)
		case int64:
			event.Int64(key, v)
		case float64:
			event.Float64(key, v)
		case bool:
			event.Bool(key, v)
		case error:
			event.Err(v)
		default:
			event.Interface(key, v)
		}
	}
}

// addFieldsToContext adds key-value pairs to a zerolog context.
func (z *zerologLogger) addFieldsToContext(ctx zerolog.Context, fields ...any) {
	for i := 0; i < len(fields)-1; i += 2 {
		key := fmt.Sprintf("%v", fields[i])
		value := fields[i+1]

		switch v := value.(type) {
		case string:
			ctx = ctx.Str(key, v)
		case int:
			ctx = ctx.Int(key, v)
		case int64:
			ctx = ctx.Int64(key, v)
		case float64:
			ctx = ctx.Float64(key, v)
		case bool:
			ctx = ctx.Bool(key, v)
		case error:
			ctx = ctx.Err(v)
		default:
			ctx = ctx.Interface(key, v)
		}
	}
}

// convertLevel converts our Level type to zerolog.Level.
func (z *zerologLogger) convertLevel(level Level) zerolog.Level {
	switch level {
	case LevelDebug:
		return zerolog.DebugLevel
	case LevelInfo:
		return zerolog.InfoLevel
	case LevelWarn:
		return zerolog.WarnLevel
	case LevelError:
		return zerolog.ErrorLevel
	default:
		return zerolog.InfoLevel
	}
}

// zerologProvider implements LoggerProvider using zerolog.
type zerologProvider struct {
	logger zerolog.Logger
	level  zerolog.Level
}

// NewZerologProvider creates a new LoggerProvider using zerolog.
func NewZerologProvider(level zerolog.Level) LoggerProvider {
	logger := zerolog.New(os.Stdout).Level(level).With().Timestamp().Caller().Logger()
	return &zerologProvider{
		logger: logger,
		level:  level,
	}
}

// GetLogger implements LoggerProvider.GetLogger.
func (p *zerologProvider) GetLogger() Logger {
	return &zerologLogger{logger: p.logger}
}

// GetLoggerWithName implements LoggerProvider.GetLoggerWithName.
func (p *zerologProvider) GetLoggerWithName(name string) Logger {
	logger := p.logger.With().Str("component", name).Logger()
	return &zerologLogger{logger: logger}
}

// SetLevel implements LoggerProvider.SetLevel.
func (p *zerologProvider) SetLevel(level Level) {
	// Safe conversion with range check
	levelInt := int(level)
	if levelInt < int(zerolog.TraceLevel) || levelInt > int(zerolog.Disabled) {
		levelInt = int(zerolog.InfoLevel) // Default to info level
	}
	zerologLevel := zerolog.Level(int8(levelInt))
	p.level = zerologLevel
	zerolog.SetGlobalLevel(zerologLevel)
	p.logger = p.logger.Level(zerologLevel)
}

// GetDefaultLogger returns the default logger instance using the new interface.
func GetDefaultLogger() Logger {
	if globalProvider == nil {
		// Initialize with info level if not already set up
		globalProvider = NewZerologProvider(zerolog.InfoLevel)
	}
	return globalProvider.GetLogger()
}

// GetLoggerWithName returns a logger with a specific component name.
func GetLoggerWithName(name string) Logger {
	if globalProvider == nil {
		globalProvider = NewZerologProvider(zerolog.InfoLevel)
	}
	return globalProvider.GetLoggerWithName(name)
}

// SetLoggerProvider sets the global logger provider.
// This is useful for testing or using different logging backends.
func SetLoggerProvider(provider LoggerProvider) {
	globalProvider = provider
}
