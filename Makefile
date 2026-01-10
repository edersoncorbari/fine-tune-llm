NOTEBOOK_DIR=notebooks

.PHONY: clean-nb

clean-nb:
	@echo "Cleaning notebooks in $(NOTEBOOK_DIR)/"
	@for nb in $(NOTEBOOK_DIR)/*.ipynb; do \
		echo "  → $$nb"; \
		jupyter nbconvert \
			--to notebook \
			--ClearMetadataPreprocessor.enabled=True \
			--output "$$(basename $$nb)" \
			"$$nb" \
			--output-dir "$(NOTEBOOK_DIR)"; \
	done
