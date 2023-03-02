FIXTURE=https://github.com/intel/openvino-rs/raw/v0.3.3/crates/openvino/tests/fixtures/mobilenet
TODIR=$1

if [ ! -f $TODIR/mobilenet.bin ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/mobilenet.bin
fi
if [ ! -f $TODIR/mobilenet.xml ]; then
    wget --no-clobber --directory-prefix=$TODIR $FIXTURE/mobilenet.xml
fi
