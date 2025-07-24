package log

import (
	"fmt"
	"log/slog"
	"os"
)

// SetupLogger function setup logger.
func SetupLogger(loglevel string) {
	ops := slog.HandlerOptions{
		AddSource: true,
		Level:     ToLogLevel(loglevel),
		// Replace attributes to convert to CloudLogging format.
		ReplaceAttr: func(groups []string, attr slog.Attr) slog.Attr {
			switch attr.Key {
			case slog.LevelKey:
				attr = slog.Attr{
					Key:   "severity",
					Value: attr.Value,
				}
			case slog.MessageKey:
				attr = slog.Attr{
					Key:   "message",
					Value: attr.Value,
				}
			case slog.SourceKey:
				attr = slog.Attr{
					Key:   "logging.googleapis.com/sourceLocation",
					Value: attr.Value,
				}
			}
			return attr
		},
	}
	handler := slog.NewJSONHandler(os.Stdout, &ops)
	errFmtHandler := WrapByErrFmtHandler(handler)
	slog.SetDefault(slog.New(errFmtHandler))
}

func ToLogLevel(level string) slog.Level {
	switch level {
	case "info":
		return slog.LevelInfo
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		panic(fmt.Sprintf("invalid log level :%s", level))
	}
}

const (
	ErrAttrKey        = "error"
	StacktraceAttrKey = "stacktrace"
)

// ErrAttr is a wrapper to pass err to slog.
func ErrAttr(err error) slog.Attr {
	return slog.Any(ErrAttrKey, err)
}