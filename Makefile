.PHONY: all test test-race bench generate vet check-license clean

all: check-license vet test

check-license:
	@echo "Checking license headers..."
	@missing=0; \
	for f in $$(git ls-files '*.go'); do \
		if ! grep -q "Licensed under the Apache License, Version 2.0" "$$f" || ! grep -q "Copyright 2026 James Grant" "$$f"; then \
			echo "Missing Apache 2.0 license header: $$f"; \
			missing=1; \
		fi; \
	done; \
	if [ $$missing -ne 0 ]; then \
		echo "License header check failed."; \
		exit 1; \
	fi; \
	echo "All Go source files contain valid license headers."

test:
	go test -v ./...

test-race:
	go test -v -race ./...

bench:
	go test -v -bench=. -benchmem ./...

generate:
	go generate ./...

vet:
	go vet ./...

clean:
	go clean -cache -testcache
