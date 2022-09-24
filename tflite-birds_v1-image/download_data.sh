FIXTURE=https://raw.githubusercontent.com/second-state/wasm-learning/master/rust/birds_v1/
TODIR=$1

if [ ! -f $TODIR/bird.jpg ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/bird.jpg
fi
if [ ! -f $TODIR/lite-model_aiy_vision_classifier_birds_V1_3.tflite ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/lite-model_aiy_vision_classifier_birds_V1_3.tflite
fi
