
JALAN:=0
SCUD2QUERY:=0
all: corpus_scud train_scud eval_scud \
	corpus_scorer train_scorer eval_scorer

.PHONY: all 
.DELETE_ON_ERROR:
SHELL=/bin/bash

ASDC_DIR:=~/data/asdc
SCUD_INTERNAL_ROOT_DIR:=~/data/scud_internal
SCUD_HOTEL_REVIEW_ROOT_DIR:=~/data/hotel_review_scud
SCUD_2_QUERY_ROOT_DIR:=~/data/scud2query
OUTPUT:=/please/designate
CONTEXT:=9999

EPOCH:=20
PERIOD:=$(EPOCH)
BATCH:=40
BATCH_DEV:=$(BATCH)
BATCH_PRED:=$(BATCH)
T5BASE:=megagonlabs/t5-base-japanese-web-8k

TRAIN_OPTION:=
OPTIMIZER:=fairseq.optim.adafactor.Adafactor
LR:=1e-3

IN_LEN:=128
OUT_LEN:=64
PREDICT_IN_LEN:=$(IN_LEN)
PREDICT_OUT_LEN:=$(OUT_LEN)
PREDICT_OPTION:=

DATA_DIR:=$(OUTPUT)/data
OUT_MODEL_DIR:=$(OUTPUT)/model
OUT_LOG:=$(OUTPUT)/logs

DATA_SCUD_DIR:=$(DATA_DIR)/scud

ifeq ($(JALAN),1)
    DATA_SCUDS_MAIN_DIR:=$(DATA_SCUD_DIR)/jalan
    DATA_SCUDS_SUP_DIR:=/dev/null
    SCUD_DIR:=$(SCUD_HOTEL_REVIEW_ROOT_DIR)/data/scud/jalan
    SCUDS_EXAMPLES:=$(shell find $(SCUD_DIR) -type f )
    CORRECTNESS_LABELED_SCUD_DIR:=$(SCUD_HOTEL_REVIEW_ROOT_DIR)/data/correctness_labeled_scud/jalan
else ifeq ($(SCUD2QUERY),1)
    DATA_SCUDS_MAIN_DIR:=$(DATA_SCUD_DIR)/scud2query
    DATA_SCUDS_SUP_DIR:=/dev/null
    SCUD_DIR:=$(SCUD_2_QUERY_ROOT_DIR)/data/scud/scud2query
    SCUDS_EXAMPLES:=$(shell find $(SCUD_DIR) -type f )
    CORRECTNESS_LABELED_SCUD_DIR:=$(SCUD_2_QUERY_ROOT_DIR)/data/correctness_labeled_scud/scud2query
    CONTEXT:=0
else
    DATA_SCUDS_MAIN_DIR:=$(DATA_SCUD_DIR)/main
    DATA_SCUDS_SUP_DIR:=$(DATA_SCUD_DIR)/sup
    ASDC_MAIN_SCUD_DIR:=$(ASDC_DIR)/data/main/scud_example
    ASDC_SUP_SCUD_DIR:=$(ASDC_DIR)/data/supplemental/scud
    SCUDS_EXAMPLES:=$(shell find $(ASDC_SUP_SCUD_DIR) $(ASDC_MAIN_SCUD_DIR) -type f )
    CORRECTNESS_LABELED_SCUD_DIR:=$(ASDC_DIR)/data/supplemental/correctness_labeled_scud

    INTERNAL_SCUD_DIR:=$(SCUD_INTERNAL_ROOT_DIR)/data/scud/for_tim
    INTERNAL_SCUDS_EXAMPLES:=$(shell find $(INTERNAL_SCUD_DIR) -type f )
    INTERNAL_CORRECTNESS_LABELED_SCUD_DIR:=$(SCUD_INTERNAL_ROOT_DIR)/data/correctness_labeled_scud/for_tim
    INTERNAL_CORRECTNESS_LABELED_SCUDS:=$(shell find $(INTERNAL_CORRECTNESS_LABELED_SCUD_DIR) -type f)
endif
CORRECTNESS_LABELED_SCUDS:=$(shell find $(CORRECTNESS_LABELED_SCUD_DIR) -type f)

NEG:=0
ifeq ($(NEG),1)
    DATA_SCUDS_SUP_DIR:=$(DATA_SCUD_DIR)/scud_negative
    TRAIN_SCUDS_NEG_GOLD:=$(DATA_SCUDS_SUP_DIR)/train.tsv
endif

TRAIN_SCUDS_MAIN_GOLD_JSONL:=$(DATA_SCUDS_MAIN_DIR)/train.jsonl
TEST_SCUDS_MAIN_GOLD_JSONL:=$(DATA_SCUDS_MAIN_DIR)/test.jsonl
DEV_SCUDS_MAIN_GOLD_JSONL:=$(DATA_SCUDS_MAIN_DIR)/dev.jsonl
DEV_SCUDS_MAIN_GOLD:=$(DATA_SCUDS_MAIN_DIR)/dev.tsv
TRAIN_SCUDS_MAIN_GOLD:=$(DATA_SCUDS_MAIN_DIR)/train.tsv
TEST_SCUDS_MAIN_GOLD:=$(DATA_SCUDS_MAIN_DIR)/test.tsv
PILOTA_CONFIG:=$(DATA_SCUDS_MAIN_DIR)/pilota.config.json

TRAIN_SCUDS_SUP_GOLD_JSONL:=$(DATA_SCUDS_SUP_DIR)/train.jsonl
TEST_SCUDS_SUP_GOLD_JSONL:=$(DATA_SCUDS_SUP_DIR)/test.jsonl
DEV_SCUDS_SUP_GOLD_JSONL:=$(DATA_SCUDS_SUP_DIR)/dev.jsonl
DEV_SCUDS_SUP_GOLD:=$(DATA_SCUDS_SUP_DIR)/dev.tsv
TRAIN_SCUDS_SUP_GOLD:=$(DATA_SCUDS_SUP_DIR)/train.tsv
TEST_SCUDS_SUP_GOLD:=$(DATA_SCUDS_SUP_DIR)/test.tsv

ifeq ($(JALAN),1)
    TRAIN_SCUDS_INPUT:=$(TRAIN_SCUDS_MAIN_GOLD)
    DEV_SCUDS_INPUT:=$(DEV_SCUDS_MAIN_GOLD)
    SCUD_TRAIN_ACCELERATOR_OPTION:=
else ifeq ($(SCUD2QUERY),1)
    TRAIN_SCUDS_INPUT:=$(TRAIN_SCUDS_MAIN_GOLD)
    DEV_SCUDS_INPUT:=$(DEV_SCUDS_MAIN_GOLD)
    SCUD_TRAIN_ACCELERATOR_OPTION:=
else
    TRAIN_SCUDS_INPUT:=$(TRAIN_SCUDS_MAIN_GOLD) $(TRAIN_SCUDS_SUP_GOLD)
    DEV_SCUDS_INPUT:=$(DEV_SCUDS_MAIN_GOLD) $(DEV_SCUDS_SUP_GOLD)
    SCUD_TRAIN_ACCELERATOR_OPTION:=
endif
SCORER_TRAIN_ACCELERATOR_OPTION:= --trainer.accelerator cuda --trainer.num_nodes 1
#CUDA_VISIBLE_DEVICES=0
ENV_PILOTA_EVAL:=
OUT_MODEL_SCUD:=$(OUT_MODEL_DIR)/scud
OUT_MODEL_WORK:=$(OUT_MODEL_SCUD).work

#------
OUT_MODEL_SCORER:=$(OUT_MODEL_DIR)/scorer

DATA_SCORER_DIR:=$(DATA_DIR)/scorer
TRAIN_SCORER_GOLD:=$(DATA_SCORER_DIR)/train.tsv
DEV_SCORER_GOLD:=$(DATA_SCORER_DIR)/dev.tsv
TEST_SCORER_GOLD:=$(DATA_SCORER_DIR)/test.tsv
LABELS_SCORER_GOLD:=$(DATA_SCORER_DIR)/labels.txt
SCORER_CONFIG:=$(DATA_SCORER_DIR)/scorer.config.json
OUT_EVAL_DIR:=$(OUTPUT)/eval
SCORER_DIR_EVAL:=$(OUT_EVAL_DIR)/scorer
SCORER_EVAL_PRED:= $(SCORER_DIR_EVAL)/prediction.jsonl
SCORER_EVAL_PRED_RESULT:= $(SCORER_DIR_EVAL)/result/done
SCORER_BATCH:=130
SCORER_EPOCH:=$(EPOCH)
SCORER_IN_LEN:=256

#------
$(OUT_MODEL_SCUD): $(TRAIN_SCUDS_INPUT) $(DEV_SCUDS_INPUT)
	mkdir -p "$(OUT_LOG)"
	mkdir -p "$(OUT_LOG)/scud"
	git -C $(ASDC_DIR) rev-parse HEAD > "$(OUT_LOG)/version.asdc.txt"
	git rev-parse HEAD > "$(OUT_LOG)/version.pilota.txt"
	mkdir -p $(OUT_MODEL_WORK)
	cp $(PILOTA_CONFIG) $(OUT_MODEL_WORK)
	time python3 -m pilota.train \
		fit \
		--data.base $(T5BASE) \
		$(addprefix --data.train+=,$(TRAIN_SCUDS_INPUT) ) \
		$(addprefix --data.dev+=,$(DEV_SCUDS_INPUT) ) \
		--data.il $(IN_LEN) \
		--data.ol $(OUT_LEN) \
		--output "$(OUT_MODEL_WORK)" \
		--trainer.logger.init_args.save_dir "$(OUT_LOG)/scud" \
		--trainer.max_epochs "$(EPOCH)" \
		--data.bs $(BATCH) \
		--data.bs_dev $(BATCH_DEV) \
		--optimizer $(OPTIMIZER) \
		--optimizer.init_args.lr $(LR) \
		--optimizer.init_args.relative_step False \
		--optimizer.init_args.scale_parameter False \
		--optimizer.init_args.warmup_init 0 \
		$(TRAIN_OPTION) \
		$(SCUD_TRAIN_ACCELERATOR_OPTION) \
	&& rm -rf $@ \
	&& mv $(OUT_MODEL_WORK) $@

train_scud: $(OUT_MODEL_SCUD)

#------

ifeq ($(JALAN),1)
$(TRAIN_SCUDS_MAIN_GOLD_JSONL): $(SCUDS_EXAMPLES)
	mkdir -p $(dir $@)
	python3 -m pilota.convert.split_example \
		$(addprefix -i ,$< ) \
		--train $@ \
		--dev  $(dir $@)/dev.jsonl \
		--test  $(dir $@)/test.jsonl

$(TRAIN_SCUDS_MAIN_GOLD): $(TRAIN_SCUDS_MAIN_GOLD_JSONL)
	python3 -m pilota.convert.example2request \
		--tsv \
		-i $(DATA_SCUDS_MAIN_DIR) \
		-o $(DATA_SCUDS_MAIN_DIR) \
		--output_config $(DATA_SCUDS_MAIN_DIR)/pilota.config.json \
		--context $(CONTEXT) \
		--name user

corpus_scud: $(TRAIN_SCUDS_MAIN_GOLD)
TEST_TARGET_FILES:= $(TEST_SCUDS_MAIN_GOLD)

else ifeq ($(SCUD2QUERY),1)
$(TRAIN_SCUDS_MAIN_GOLD_JSONL): $(SCUDS_EXAMPLES)
	mkdir -p $(dir $@)
	python3 -m pilota.convert.split_example \
		$(addprefix -i ,$< ) \
		--train $@ \
		--dev  $(dir $@)/dev.jsonl \
		--test  $(dir $@)/test.jsonl

$(TRAIN_SCUDS_MAIN_GOLD): $(TRAIN_SCUDS_MAIN_GOLD_JSONL)
	python3 -m pilota.convert.example2request \
		--tsv \
		-i $(DATA_SCUDS_MAIN_DIR) \
		-o $(DATA_SCUDS_MAIN_DIR) \
		--output_config $(DATA_SCUDS_MAIN_DIR)/pilota.config.json \
		--context $(CONTEXT) \
		--name user

corpus_scud: $(TRAIN_SCUDS_MAIN_GOLD)
TEST_TARGET_FILES:= $(TEST_SCUDS_MAIN_GOLD)
else
$(TRAIN_SCUDS_MAIN_GOLD_JSONL): $(SCUDS_EXAMPLES) $(INTERNAL_SCUDS_EXAMPLES)
	find $(SCUD_INTERNAL_ROOT_DIR)/data/scud/for_tim/*.Example.jsonl | xargs -n1 test -r
	mkdir -p "$(OUT_LOG)"
	git -C $(ASDC_DIR) rev-parse HEAD > "$(OUT_LOG)/version.scud_internal.txt"
	$(MAKE) -C "$(ASDC_DIR)" -f ./mks/generate_example.mk \
		ROOT_DIR=$(ASDC_DIR) \
		generate_example_main \
		generate_example_sup \
		OUTPUT=$(DATA_SCUD_DIR) \
		INPUT_SUP_EXTRA='$(SCUD_INTERNAL_ROOT_DIR)/data/scud/for_tim/*.Example.jsonl'
$(TRAIN_SCUDS_SUP_GOLD_JSONL): $(TRAIN_SCUDS_MAIN_GOLD_JSONL)

$(TRAIN_SCUDS_MAIN_GOLD): $(TRAIN_SCUDS_MAIN_GOLD_JSONL)
	python3 -m pilota.convert.example2request \
		--tsv \
		-i $(DATA_SCUDS_MAIN_DIR) \
		-o $(DATA_SCUDS_MAIN_DIR) \
		--output_config $(DATA_SCUDS_MAIN_DIR)/pilota.config.json \
		--context $(CONTEXT) \
		--name agent --name user

$(TRAIN_SCUDS_SUP_GOLD): $(TRAIN_SCUDS_SUP_GOLD_JSONL)
	python3 -m pilota.convert.example2request \
		--tsv \
		-i $(DATA_SCUDS_SUP_DIR) \
		-o $(DATA_SCUDS_SUP_DIR) \
		--context $(CONTEXT) \
		--output_config $(DATA_SCUDS_SUP_DIR)/pilota.config.json \
		--name agent --name user

corpus_scud: $(TRAIN_SCUDS_MAIN_GOLD) $(TRAIN_SCUDS_SUP_GOLD)

$(DEV_SCUDS_SUP_GOLD): $(TRAIN_SCUDS_SUP_GOLD)
$(TEST_SCUDS_SUP_GOLD): $(TRAIN_SCUDS_SUP_GOLD)
TEST_TARGET_FILES:= $(TEST_SCUDS_MAIN_GOLD) $(TEST_SCUDS_SUP_GOLD)
$(TEST_SCUDS_SUP_GOLD_JSONL) $(DEV_SCUDS_SUP_GOLD_JSONL): $(TRAIN_SCUDS_SUP_GOLD_JSONL)

endif
$(TEST_SCUDS_MAIN_GOLD_JSONL) $(DEV_SCUDS_MAIN_GOLD_JSONL): $(TRAIN_SCUDS_MAIN_GOLD_JSONL)
$(DEV_SCUDS_MAIN_GOLD): $(TRAIN_SCUDS_MAIN_GOLD)
$(TEST_SCUDS_MAIN_GOLD): $(TRAIN_SCUDS_MAIN_GOLD)



OUT_PRED_SCUD_DIR:=$(OUT_EVAL_DIR)/scud
SUFFIX_PREDICTION:=.prediction.jsonl
SUFFIX_EVAL:=.eval.jsonl
SUFFIX_STAT:=.eval.stat.tsv
SUFFIX_CSV:=.eval.csv

TEST_PREDICTION_FILES:=$(patsubst $(DATA_SCUD_DIR)%.tsv,$(OUT_PRED_SCUD_DIR)%$(SUFFIX_PREDICTION),$(TEST_TARGET_FILES))
TEST_EVAL_FILES:=$(patsubst $(DATA_SCUD_DIR)%.tsv,$(OUT_PRED_SCUD_DIR)%$(SUFFIX_EVAL),$(TEST_TARGET_FILES))
TEST_STAT_FILES:=$(patsubst $(DATA_SCUD_DIR)%.tsv,$(OUT_PRED_SCUD_DIR)%$(SUFFIX_STAT),$(TEST_TARGET_FILES))
TEST_CSV_FILES:=$(patsubst $(DATA_SCUD_DIR)%.tsv,$(OUT_PRED_SCUD_DIR)%$(SUFFIX_CSV),$(TEST_TARGET_FILES))

eval_scud: $(TEST_PREDICTION_FILES) $(TEST_EVAL_FILES) $(TEST_STAT_FILES) $(TEST_CSV_FILES)



IN_PRED_JOINT:=$(OUT_PRED_SCUD_DIR)/test.joint.txt
OUT_PRED_JOINT:=$(OUT_PRED_SCUD_DIR)/test.joint$(SUFFIX_PREDICTION)
$(IN_PRED_JOINT): $(TEST_TARGET_FILES)
	mkdir -p $(dir $@) && \
	python3 -m pilota.evaluate.joint $(addprefix --input ,$(TEST_TARGET_FILES) ) -o $@.tmp \
		&& rm -rf $@ \
		&& mv $@.tmp $@

$(OUT_PRED_JOINT): $(IN_PRED_JOINT) $(OUT_MODEL_SCUD) $(OUT_MODEL_SCORER)
	mkdir -p $(dir $@) && \
	cut -f2 $(IN_PRED_JOINT) \
		| $(ENV_PILOTA_EVAL) python3 -m pilota.cli --raw_in \
		--model $(OUT_MODEL_DIR) -o $@ \
		$(NO_SCORER_OPTION) \
		--il $(PREDICT_IN_LEN) --ol $(PREDICT_OUT_LEN) --bs $(BATCH_PRED) $(PREDICT_OPTION)

$(OUT_PRED_SCUD_DIR)/%$(SUFFIX_PREDICTION): $(DATA_SCUD_DIR)/%.tsv $(OUT_PRED_JOINT)
	mkdir -p $(dir $@) && \
	python3 -m pilota.evaluate.joint --ref $(IN_PRED_JOINT) -i $(OUT_PRED_JOINT) -s $(DATA_SCUD_DIR)/$*.jsonl -o $@

$(OUT_PRED_SCUD_DIR)/%$(SUFFIX_EVAL): $(OUT_PRED_SCUD_DIR)/%$(SUFFIX_PREDICTION) $(DATA_SCUD_DIR)/%.jsonl
	mkdir -p $(dir $@) && \
	python3 -m pilota.evaluate.cli \
		-g $(DATA_SCUD_DIR)/$*.tsv \
		-i $(OUT_PRED_SCUD_DIR)/$*$(SUFFIX_PREDICTION) \
		--txt $(OUT_PRED_SCUD_DIR)/txt/$* \
		--max_nbest_for_txt 4 \
		-o $@

$(OUT_PRED_SCUD_DIR)/%$(SUFFIX_STAT): $(OUT_PRED_SCUD_DIR)/%$(SUFFIX_EVAL)
	mkdir -p $(dir $@) && \
	python3 -m pilota.evaluate.cli --stat -i $< -o $@

$(OUT_PRED_SCUD_DIR)/%$(SUFFIX_CSV): $(OUT_PRED_SCUD_DIR)/%$(SUFFIX_EVAL)
	mkdir -p $(dir $@) && \
	python3 -m pilota.evaluate.cli --csv -i $< -o $@

#--------


ifeq ($(JALAN),1)
$(DATA_SCORER_DIR): $(CORRECTNESS_LABELED_SCUDS) $(SCUDS_EXAMPLES)
	python -m pilota.convert.scorer \
		-i $(CORRECTNESS_LABELED_SCUD_DIR) \
		--original $(SCUD_DIR) \
		--context $(CONTEXT) \
		--name user \
		-o $@.tmp \
		--extra all \
		&& rm -rf $@ \
		&& mv $@.tmp $@
else ifeq ($(SCUD2QUERY),1)
$(DATA_SCORER_DIR): $(CORRECTNESS_LABELED_SCUDS) $(SCUDS_EXAMPLES)
	python -m pilota.convert.scorer \
		-i $(CORRECTNESS_LABELED_SCUD_DIR) \
		--original $(SCUD_DIR) \
		--context $(CONTEXT) \
		--name user \
		-o $@.tmp \
		--extra all \
		&& rm -rf $@ \
		&& mv $@.tmp $@
else

$(DATA_SCORER_DIR): $(CORRECTNESS_LABELED_SCUDS) $(SCUDS_EXAMPLES) $(INTERNAL_CORRECTNESS_LABELED_SCUDS) $(INTERNAL_SCUDS_EXAMPLES)
	python -m pilota.convert.scorer \
		-i $(CORRECTNESS_LABELED_SCUD_DIR) \
		--original $(ASDC_SUP_SCUD_DIR)  \
		-i $(INTERNAL_CORRECTNESS_LABELED_SCUD_DIR) \
		--original $(INTERNAL_SCUD_DIR) \
		--context $(CONTEXT) \
		--context_separator "<INPUT>" \
		--name agent --name user \
		-o $@.tmp \
		&& rm -rf $@ \
		&& mv $@.tmp $@
endif

$(TRAIN_SCORER_GOLD) $(DEV_SCORER_GOLD) $(TEST_SCORER_GOLD) $(LABELS_SCORER_GOLD): $(DATA_SCORER_DIR)
corpus_scorer: $(DATA_SCORER_DIR)

SCORER_BASE:=line-corporation/line-distilbert-base-japanese
SCORER_LR:=1e-4
$(OUT_MODEL_SCORER): $(TRAIN_SCORER_GOLD) $(DEV_SCORER_GOLD) $(LABELS_SCORER_GOLD)
	mkdir -p "$(OUT_LOG)/scorer"
	mkdir -p $(dir $@) && \
	eval python -m pilota.mull.train \
		fit \
		--data.base $(SCORER_BASE) \
		--data.train $(TRAIN_SCORER_GOLD)\
		--data.dev $(DEV_SCORER_GOLD) \
		--data.label $(LABELS_SCORER_GOLD) \
		--data.il $(SCORER_IN_LEN) \
		--output $@.tmp \
		--trainer.max_epochs "$(SCORER_EPOCH)" \
		--data.bs $(SCORER_BATCH) \
		--data.bs_dev $(SCORER_BATCH) \
		--trainer.logger.init_args.save_dir "$(OUT_LOG)/scorer" \
		--optimizer AdamW \
		--optimizer.init_args.weight_decay 0.0001 \
		--optimizer.init_args.lr $(SCORER_LR) \
		$(SCORER_TRAIN_ACCELERATOR_OPTION) \
	&& cp "$(SCORER_CONFIG)" $@.tmp \
	&& rm -rf $@ \
	&& mv $@.tmp $@

$(SCORER_EVAL_PRED) :$(OUT_MODEL_SCORER) $(TEST_SCORER_GOLD)
	mkdir -p $(dir $@) && \
	cut -f2 $(TEST_SCORER_GOLD) \
		| python -m pilota.scorer -m $(OUT_MODEL_SCORER) \
		> $@.tmp \
	&& rm -rf $@ \
	&& mv $@.tmp $@

$(SCORER_EVAL_PRED_RESULT): $(SCORER_EVAL_PRED) $(TEST_SCORER_GOLD) $(LABELS_SCORER_GOLD)
	mkdir -p $(dir $@) \
	&& python -m pilota.evaluate.scorer -i $(SCORER_EVAL_PRED) -g $(TEST_SCORER_GOLD) -o $(dir $@) --label $(LABELS_SCORER_GOLD) \
	&& touch $@

train_scorer: $(OUT_MODEL_SCORER)
eval_scorer: $(SCORER_EVAL_PRED) $(SCORER_EVAL_PRED_RESULT)

scud: corpus_scud train_scud eval_scud
scorer: corpus_scorer train_scorer eval_scorer

#--------
.PHONY: all \
	corpus_scud train_scud eval_scud \
	corpus_scorer train_scorer eval_scorer
