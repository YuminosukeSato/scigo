package log

import (
	"context"
	"log/slog"

	"github.com/cockroachdb/errors"
)

// ErrFmtHandler is a slog handler to format stacktrace from cockroachdb/errors.
type ErrFmtHandler struct {
	handler slog.Handler
}

// WrapByErrFmtHandler function wraps the standard slog handler.
// This function returns the slog handler which emits logs with a stacktrace attribute.
func WrapByErrFmtHandler(handler slog.Handler) slog.Handler {
	return &ErrFmtHandler{
		handler: handler,
	}
}

func (eh *ErrFmtHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return eh.handler.Enabled(ctx, l)
}

func (eh *ErrFmtHandler) Handle(ctx context.Context, r slog.Record) error {
	var stacktrace string
	r.Attrs(func(attr slog.Attr) bool {
		if attr.Key == ErrAttrKey {
			err, ok := attr.Value.Any().(error)
			if ok {
				stacktrace = extractStacktrace(err)
			}
			return false
		}
		return true
	})
	if stacktrace != "" {
		r.AddAttrs(slog.String(StacktraceAttrKey, stacktrace))
	}
	return eh.handler.Handle(ctx, r)
}

func (eh *ErrFmtHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	return &ErrFmtHandler{handler: eh.handler.WithAttrs(attrs)}
}

func (eh *ErrFmtHandler) WithGroup(g string) slog.Handler {
	return &ErrFmtHandler{handler: eh.handler.WithGroup(g)}
}

func extractStacktrace(err error) string {
	safeDetails := errors.GetSafeDetails(err).SafeDetails
	if len(safeDetails) > 0 {
		return safeDetails[0]
	}
	return ""
}