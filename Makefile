.PHONY: test bench fmt lint build all clean

# デフォルトターゲット
all: fmt lint test

# テスト実行
test:
	go test -v ./...

# ベンチマーク実行
bench:
	go test -bench=. -benchmem ./...

# コードフォーマット
fmt:
	go fmt ./...

# 静的解析
lint:
	go vet ./...

# ビルド（全パッケージのコンパイルチェック）
build:
	go build -v ./...

# クリーンアップ
clean:
	go clean -cache
	go mod tidy

# カバレッジ測定
coverage:
	go test -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

# 依存関係の更新
deps:
	go mod download
	go mod tidy

# ドキュメント生成（ローカルサーバー起動）
docs:
	@echo "Starting Go documentation server at http://localhost:6060"
	godoc -http=:6060