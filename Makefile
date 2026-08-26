.PHONY: all test test-race bench generate vet clean

all: vet test

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
