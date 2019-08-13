.PHONY: pkg
pkg:
	python3 setup.py sdist bdist_wheel

.PHONY: upload
upload:
	twine upload dist/*
