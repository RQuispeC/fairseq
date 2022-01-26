# datapaths
train_document=/data/msft/msftrodolfo_quispe/data/xsum/train.document
train_summary=/data/msft/msftrodolfo_quispe/data/xsum/train.summary
validation_document=/data/msft/msftrodolfo_quispe/data/xsum/validation.document
validation_summary=/data/msft/msftrodolfo_quispe/data/xsum/validation.summary
test_document=/data/msft/msftrodolfo_quispe/data/xsum/test.document

train_src=train.pg.src
train_tgt=train.pg.tgt
validation_src=valid.pg.src
validation_tgt=valid.pg.tgt
test_src=test.pg.src
test_tgt=test.pg.tgt


# create dictionary
echo "creating dictionary"
vocab_size=10000
position_markers=1000
export LC_ALL=C
cat $train_document $train_summary |
  tr -s '[:space:]' '\n' |
  sort |
  uniq -c |
  sort -k1,1bnr -k2 |
  head -n "$((vocab_size - 4))" |
  awk '{ print $2 " " $1 }' >dict.pg.txt
python3 -c "[print('<unk-{}> 0'.format(n)) for n in range($position_markers)]" >>dict.pg.txt

# preprocess data
echo "preprocessing data"
echo "trainset ..."
./preprocess.py --source $train_document --target $train_summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out $train_src --target-out $train_tgt
echo "validation set ..."
./preprocess.py --source $validation_document --target $validation_summary --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out $validation_src --target-out $validation_tgt
echo "test set ..."
./preprocess.py --source $test_document --vocab <(cut -d' ' -f1 dict.pg.txt) --source-out $test_src

# binarize data
echo "binarizing data"
fairseq-preprocess \
  --source-lang src \
  --target-lang tgt \
  --trainpref train.pg \
  --validpref valid.pg \
  --destdir bin \
  --workers 60 \
  --srcdict dict.pg.txt \
  --joined-dictionary

