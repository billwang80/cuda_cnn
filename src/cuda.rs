use crate::cnn::*;
use rustacuda::function::BlockSize;
use rustacuda::launch;
use rustacuda::memory::DeviceCopy;
use rustacuda::memory::DeviceBox;
use rustacuda::prelude::*;
use std::error::Error;
use std::ffi::CString;

// Fields need to be ordered this way so the DeviceBoxes are
// dropped before the Context. Otherwise the drop will panic.

pub struct CudaContext {
    conv_layer: DeviceBox<ConvLayer>,
    output_layer: DeviceBox<OutputLayer>,
    module: Module,
    stream: Stream,
    _context: Context,
}

impl CudaContext {
    pub fn init(cnn: &Cnn) -> Result<Self, Box<dyn Error>> {
        rustacuda::init(CudaFlags::empty())?;
        let device = Device::get_device(0)?;
        let context = Context::create_and_push(ContextFlags::MAP_HOST | ContextFlags::SCHED_AUTO, device)?;

        let ptx = CString::new(include_str!("../kernel/kernel.ptx"))?;
        let module = Module::load_from_string(&ptx)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

        let conv_layer = DeviceBox::new(&cnn.conv_layer).unwrap();
        let output_layer = DeviceBox::new(&cnn.output_layer).unwrap();

        Ok(CudaContext {
            conv_layer,
            output_layer,
            module,
            stream,
            _context: context,
        })
    }

    pub fn compute(&mut self, input: &InputMatrix) -> Result<OutputVec, Box<dyn Error>> {
        let mut input_box = DeviceBox::new(input).unwrap();
        let input_ptr = input_box.as_device_ptr();
        
        let conv_layer_ptr = self.conv_layer.as_device_ptr();
        let output_layer_ptr = self.output_layer.as_device_ptr();

        let mut conv_output = DeviceBox::new(&ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]))?;
        let conv_output_ptr = conv_output.as_device_ptr();

        let module = &self.module;
        let stream = &self.stream;

        unsafe {
            launch!(module.convolution<<<(20,20),10,0,stream>>>(
                input_ptr,
                conv_layer_ptr,
                conv_output_ptr
            ))?;
        }
        stream.synchronize()?;

        unsafe {
            launch!(module.relu<<<(20,20),10,0,stream>>>(
                conv_output_ptr
            ))?;
        }
        stream.synchronize()?;

        let mut output_box = DeviceBox::new(&OutputVec([0.0; OUT_LAYER_SIZE]))?;
        let output_ptr = output_box.as_device_ptr();

        unsafe {
            launch!(module.output<<<1,10,0,stream>>>(
                conv_output_ptr,
                output_layer_ptr,
                output_ptr
            ))?;
        }
        stream.synchronize()?;

        let mut res = ConvOutput([[[0.0; CONV_OUT_DIM]; CONV_OUT_DIM]; CONV_LAYER_SIZE]);
        conv_output.copy_to(&mut res)?;

        let mut result = OutputVec([0.0; OUT_LAYER_SIZE]);
        output_box.copy_to(&mut result)?;
        Ok(result)
    }
}
