
all: lint lint_markdown test

flake8:
	find ./pilota ./tests | grep '\.py$$' | xargs flake8
black:
	find ./pilota ./tests | grep '\.py$$' | xargs black --diff | diff /dev/null -
pyright:
	npx pyright
isort:
	find ./pilota ./tests | grep '\.py$$' | xargs isort --diff | diff /dev/null -
pydocstyle:
	find ./pilota ./tests | grep '\.py$$' | grep -v run_clm.py | \
		xargs pydocstyle --ignore=D100,D101,D102,D103,D104,D105,D107,D203,D212,D400,D415

jsonlint:
	find .*json ./pilota ./tests -type f | grep '\.jsonl$$' | sort |xargs cat | python3 -c 'import sys,json; [json.loads(line) for line in sys.stdin]'
	find .*json ./pilota ./tests -type f | grep '\.json$$' | sort |xargs -n 1 -t python3 -m json.tool > /dev/null
	python3 -c "import sys,json;print(json.dumps(json.loads(sys.stdin.read()),indent=4,ensure_ascii=False,sort_keys=True))" < .markdownlint.json | diff -q - .markdownlint.json

yamllint:
	find .github -name '*.yml' -type f | grep -v node_modules | xargs yamllint --no-warnings

lint: flake8 black pyright isort yamllint pydocstyle lint_cff

_run_isort:
	isort -rc .

_test:
	coverage run -m unittest discover tests

test: _test _coverage

test-coverage: test
	coverage report && coverage html

CC_REPORTER_VERSION:=latest
setup-cc:
	mkdir -p ~/.local/bin-cc
	curl -L https://codeclimate.com/downloads/test-reporter/test-reporter-$(CC_REPORTER_VERSION)-linux-amd64 > ~/.local/bin-cc/cc-test-reporter
	chmod +x ~/.local/bin-cc/cc-test-reporter
	~/.local/bin-cc/cc-test-reporter before-build

test-cc: test
	coverage xml && \
	 ~/.local/bin-cc/cc-test-reporter after-build\
	 --coverage-input-type coverage.py\
	 --exit-code $$?

setup_node_module:
	npm install markdownlint-cli

lint_markdown:
	find . -type d -o -type f -name '*.md' -print \
	| grep -v \.venv \
	| grep -v node_modules \
	| xargs npx markdownlint --config ./.markdownlint.json

lint_cff:
	cffconvert --validate

.PHONY: all setup \
	flake8 black pyright isort jsonlint yamllint\
	check_firstline \
	lint \
	_run_isort _test _coverage\
	test test-coverage setup-cc test-cc\
	setup_node_module lint_markdown lint_cff

.DELETE_ON_ERROR:
