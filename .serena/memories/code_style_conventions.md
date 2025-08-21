# Code Style and Conventions

## Import Ordering
Imports must be organized in three groups separated by blank lines:
1. Standard library imports
2. Third-party imports (e.g., gonum.org/v1/gonum/mat)
3. Local project imports (github.com/YuminosukeSato/scigo/*)

Use `goimports -w -local github.com/YuminosukeSato/scigo .` to fix import ordering.

## Go Conventions
- Package names: lowercase, single word
- Exported functions/types: PascalCase
- Unexported functions/types: camelCase
- Error handling: Always check errors, use errors.Is/As
- Comments: Exported items must have godoc comments starting with the item name
- Testing: Use testify for assertions, table-driven tests preferred

## File Organization
- One main type per file
- Test files alongside implementation files
- Example tests in example_test.go files

## Error Handling
- Use github.com/YuminosukeSato/scigo/pkg/errors for error wrapping
- Always wrap errors with context using fmt.Errorf or errors.Wrap
- Check errors immediately after function calls

## Logging
- Use github.com/YuminosukeSato/scigo/pkg/log for logging
- Structured logging with zerolog under the hood
- Log levels: Debug, Info, Warn, Error