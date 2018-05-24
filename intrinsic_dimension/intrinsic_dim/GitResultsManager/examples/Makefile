all : demo-c run-demos

demo-c : demo-c.cc
	g++ -o $@ $<

run-demos :
	@echo
	@echo "Running demo: demo-c"
	gitresman junk ./demo-c
	@echo
	@echo "Running demo: demo-python.py "
	gitresman junk ./demo-python.py
	@echo
	@echo "Running demo: demo-python-with-flush.py "
	gitresman junk ./demo-python-with-flush.py
	@echo
	@echo "Running demo: demo-GRM-module.py"
	./demo-GRM-module.py
