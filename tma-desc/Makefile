all: main test_bit21_oob

main: main.cc cuTensorMapEncodeTiled_impl.h
	nvcc -o $@ $< -std=c++17 -arch=sm_100a -lcuda

test_bit21_oob: test_bit21_oob.cu
	nvcc -gencode arch=compute_100a,code=sm_100a -o $@ $< -lcuda

clean:
	rm -f main test_bit21_oob

test: main
	@pass=0; fail=0; skip=0; total=0; \
	while IFS= read -r line; do \
		case "$$line" in \#*|"") continue;; esac; \
		total=$$((total + 1)); \
		output=$$(./main $$line 2>/dev/null); \
		if echo "$$output" | grep -q "\\[PASS\\]"; then \
			pass=$$((pass + 1)); \
		elif echo "$$output" | grep -q "\\[SKIP\\]"; then \
			skip=$$((skip + 1)); \
		else \
			fail=$$((fail + 1)); \
			echo "FAIL: $$line"; \
		fi; \
	done < tests.txt; \
	echo ""; \
	echo "Results: $$pass PASS, $$fail FAIL, $$skip SKIP ($$total total)"; \
	[ $$fail -eq 0 ]

.PHONY: all clean test
