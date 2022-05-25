FIXTURE=https://github.com/intel/openvino-rs/raw/main/crates/openvino/tests/fixtures/mobilenet
TODIR=$1

if [ ! -f $TODIR/mobilenet.bin ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/mobilenet.bin
fi
if [ ! -f $TODIR/mobilenet.xml ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/mobilenet.xml
fi
if [ ! -f $TODIR/tensor-1x224x224x3-f32.bgr ]; then
    wget --no-clobber $FIXTURE/tensor-1x224x224x3-f32.bgr --output-document=$TODIR/tensor-1x224x224x3-f32.bgr
fi
