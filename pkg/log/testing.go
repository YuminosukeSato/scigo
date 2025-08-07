// Package log provides testing utilities for structured logging.
//
// This file contains helper functions and types specifically designed for
// testing logging functionality in SciGo. It provides ways to capture and
// verify log output during tests without interfering with the normal
// execution flow.

package log

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// TestLogger is a logger implementation designed for testing.
// It captures all log messages in memory for later inspection and verification.
type TestLogger struct {
	buffer *bytes.Buffer
	level  Level
	fields map[string]interface{}
}

// NewTestLogger creates a new TestLogger with the specified minimum level.
// All log messages are captured in an internal buffer for later examination.
//
// Parameters:
//   - level: Minimum log level to capture
//
// Returns:
//   - *TestLogger: A new test logger instance
//   - *bytes.Buffer: The buffer containing captured log output
//
// Example:
//
//	logger, buffer := log.NewTestLogger(log.LevelDebug)
//	logger.Info("test message", "key", "value")
//	output := buffer.String()
//	// Verify output contains expected content
func NewTestLogger(level Level) (*TestLogger, *bytes.Buffer) {
	buffer := &bytes.Buffer{}
	return &TestLogger{
		buffer: buffer,
		level:  level,
		fields: make(map[string]interface{}),
	}, buffer
}

// Debug implements Logger.Debug.
func (t *TestLogger) Debug(msg string, fields ...any) {
	if t.level <= LevelDebug {
		t.writeLog("DEBUG", msg, fields...)
	}
}

// Info implements Logger.Info.
func (t *TestLogger) Info(msg string, fields ...any) {
	if t.level <= LevelInfo {
		t.writeLog("INFO", msg, fields...)
	}
}

// Warn implements Logger.Warn.
func (t *TestLogger) Warn(msg string, fields ...any) {
	if t.level <= LevelWarn {
		t.writeLog("WARN", msg, fields...)
	}
}

// Error implements Logger.Error.
func (t *TestLogger) Error(msg string, fields ...any) {
	if t.level <= LevelError {
		t.writeLog("ERROR", msg, fields...)
	}
}

// With implements Logger.With.
func (t *TestLogger) With(fields ...any) Logger {
	newFields := make(map[string]interface{})

	// Copy existing fields
	for k, v := range t.fields {
		newFields[k] = v
	}

	// Add new fields
	for i := 0; i < len(fields)-1; i += 2 {
		key := fmt.Sprintf("%v", fields[i])
		value := fields[i+1]

		// Handle special cases for error types
		if err, ok := value.(error); ok {
			newFields[key] = err.Error()
		} else {
			newFields[key] = value
		}
	}

	return &TestLogger{
		buffer: t.buffer,
		level:  t.level,
		fields: newFields,
	}
}

// Enabled implements Logger.Enabled.
func (t *TestLogger) Enabled(ctx context.Context, level Level) bool {
	return t.level <= level
}

// writeLog writes a log entry to the buffer in JSON format.
func (t *TestLogger) writeLog(level, msg string, fields ...any) {
	entry := map[string]interface{}{
		"level":   level,
		"message": msg,
	}

	// Add existing fields
	for k, v := range t.fields {
		entry[k] = v
	}

	// Add new fields
	for i := 0; i < len(fields)-1; i += 2 {
		key := fmt.Sprintf("%v", fields[i])
		value := fields[i+1]

		// Handle special cases for error types
		if err, ok := value.(error); ok {
			entry[key] = err.Error()
		} else {
			entry[key] = value
		}
	}

	// Write JSON line
	jsonData, _ := json.Marshal(entry)
	t.buffer.WriteString(string(jsonData) + "\n")
}

// GetBuffer returns the internal buffer for direct access to captured logs.
func (t *TestLogger) GetBuffer() *bytes.Buffer {
	return t.buffer
}

// GetLogEntries parses the captured log output and returns structured log entries.
// This is useful for programmatic verification of log content.
//
// Returns:
//   - []map[string]interface{}: Slice of parsed log entries
//   - error: Error if log parsing fails
//
// Example:
//
//	entries, err := testLogger.GetLogEntries()
//	if err != nil {
//	    t.Fatal(err)
//	}
//	if len(entries) != 2 {
//	    t.Errorf("Expected 2 log entries, got %d", len(entries))
//	}
func (t *TestLogger) GetLogEntries() ([]map[string]interface{}, error) {
	var entries []map[string]interface{}
	lines := strings.Split(strings.TrimSpace(t.buffer.String()), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}

		var entry map[string]interface{}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			return nil, err
		}
		entries = append(entries, entry)
	}

	return entries, nil
}

// ContainsMessage checks if the captured logs contain a message with the specified content.
// This is a convenience method for common test assertions.
//
// Parameters:
//   - message: The message content to search for
//
// Returns:
//   - bool: true if the message is found in any log entry
//
// Example:
//
//	if !testLogger.ContainsMessage("Training completed") {
//	    t.Error("Expected training completion log message")
//	}
func (t *TestLogger) ContainsMessage(message string) bool {
	return strings.Contains(t.buffer.String(), message)
}

// ContainsField checks if the captured logs contain an entry with the specified field and value.
//
// Parameters:
//   - key: The field key to search for
//   - value: The expected field value
//
// Returns:
//   - bool: true if the field with the specified value is found
//
// Example:
//
//	if !testLogger.ContainsField("ml.operation", "fit") {
//	    t.Error("Expected fit operation in logs")
//	}
func (t *TestLogger) ContainsField(key string, value interface{}) bool {
	entries, err := t.GetLogEntries()
	if err != nil {
		return false
	}

	for _, entry := range entries {
		if fieldValue, exists := entry[key]; exists {
			if fieldValue == value {
				return true
			}
		}
	}

	return false
}

// Clear clears all captured log content.
// Useful for resetting state between test cases.
func (t *TestLogger) Clear() {
	t.buffer.Reset()
}

// TestLoggerProvider implements LoggerProvider for testing scenarios.
type TestLoggerProvider struct {
	logger *TestLogger
	buffer *bytes.Buffer
}

// NewTestLoggerProvider creates a new test logger provider.
//
// Parameters:
//   - level: Minimum log level to capture
//
// Returns:
//   - *TestLoggerProvider: A new test provider instance
//   - *bytes.Buffer: Buffer for accessing captured logs
func NewTestLoggerProvider(level Level) (*TestLoggerProvider, *bytes.Buffer) {
	logger, buffer := NewTestLogger(level)
	return &TestLoggerProvider{
		logger: logger,
		buffer: buffer,
	}, buffer
}

// GetLogger implements LoggerProvider.GetLogger.
func (p *TestLoggerProvider) GetLogger() Logger {
	return p.logger
}

// GetLoggerWithName implements LoggerProvider.GetLoggerWithName.
func (p *TestLoggerProvider) GetLoggerWithName(name string) Logger {
	return p.logger.With("component", name)
}

// SetLevel implements LoggerProvider.SetLevel.
func (p *TestLoggerProvider) SetLevel(level Level) {
	p.logger.level = level
}

// GetBuffer returns the buffer for accessing captured logs.
func (p *TestLoggerProvider) GetBuffer() *bytes.Buffer {
	return p.buffer
}
