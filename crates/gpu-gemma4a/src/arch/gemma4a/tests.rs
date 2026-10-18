use super::*;
use bloomery_gpu::DeviceTensor;

#[test]
#[ignore = "opens the card for the clipped-linear contract"]
fn hw_clipped_linear() -> Result<(), GpuError> {
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let pass = Pass::load(&gpu)?;
    let mut linear = Linear {
        w: DeviceTensor::upload(stream, &[1.0; 64], 1, 64)?,
        clamp: [-0.5, 0.5, -10.0, 10.0],
        bf16: true,
    };
    let input: Vec<_> = [-2.0, 1.0 / 3.0, 1.0, 0.0]
        .into_iter()
        .cycle()
        .take(64)
        .collect();
    let x = DeviceBuffer::from_host(stream, &input)?;
    let mut tmp = DeviceBuffer::zeroed(stream, 64)?;
    let mut y = DeviceBuffer::zeroed(stream, 1)?;
    pass.mm(stream, &linear, &x, 1, &mut tmp, &mut y)?;
    assert_eq!(
        y.to_host_vec(stream)?,
        vec![5.34375],
        "input clamp then BF16 rounding"
    );
    linear.clamp[3] = 3.0;
    pass.mm(stream, &linear, &x, 1, &mut tmp, &mut y)?;
    assert_eq!(y.to_host_vec(stream)?, vec![3.0], "output maximum");
    linear.clamp[2] = 6.0;
    linear.clamp[3] = 8.0;
    pass.mm(stream, &linear, &x, 1, &mut tmp, &mut y)?;
    assert_eq!(y.to_host_vec(stream)?, vec![6.0], "output minimum");
    linear.clamp[2] = -f32::MAX;
    linear.clamp[3] = f32::MAX;
    pass.mm(stream, &linear, &x, 1, &mut tmp, &mut y)?;
    assert_eq!(
        y.to_host_vec(stream)?,
        vec![5.34375],
        "output bounds at the extremes"
    );
    stream.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "opens the card for the clamp launch the extreme bounds skip"]
fn hw_extreme_bounds_clamp_changes_no_bit() -> Result<(), GpuError> {
    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let pass = Pass::load(&gpu)?;
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f32::MAX,
        -f32::MAX,
        f32::MIN_POSITIVE,
        1.0e-45,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(0x7fc0_1234),
    ];
    let mut x = DeviceBuffer::from_host(stream, &values)?;
    pass.unary(stream, values.len(), [-f32::MAX, f32::MAX], 0, &mut x)?;
    let bits = |v: &[f32]| v.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&x.to_host_vec(stream)?), bits(&values));
    Ok(())
}
#[test]
#[ignore = "opens the card for audio arithmetic overflow refusals"]
fn hw_arithmetic_overflow() -> Result<(), GpuError> {
    use attention::{HEADS, RPE, WIDTH};
    use bloomery_gpu::{FaultSite, fault};

    let gpu = Gpu::new()?;
    let stream = gpu.stream();
    let pass = Pass::load(&gpu)?;
    let ops = &pass.ops;
    let attention = &pass.attention;
    let mut out = DeviceBuffer::<f32>::zeroed(stream, WIDTH)?;
    let mut refused = Vec::new();

    let gain = DeviceBuffer::from_host(stream, &vec![1.0; WIDTH])?;
    for (boundary, value) in [("RMS square", 1e20), ("RMS sum", 1e18)] {
        let x = DeviceBuffer::from_host(stream, &vec![value; WIDTH])?;
        pass.elem
            .enqueue_rms_norm(stream, &x, &gain, 1e-6, WIDTH, 1, &mut out)?;
        stream.synchronize()?;
        refused.push((boundary, fault::read(gpu.fault_word())?));
        gpu.clear_fault()?;
    }

    let gain = DeviceBuffer::from_host(stream, &[1.0, 1.0])?;
    let prep = ops.module.prepare_channel_norm(config(1))?;
    for (boundary, values) in [
        ("channel mean", [f32::MAX; 2]),
        ("channel variance", [1e20, -1e20]),
    ] {
        let x = DeviceBuffer::from_host(stream, &values)?;
        ops.module
            .channel_norm(stream, &prep, &x, &gain, 1, 2, 1e-6, ops.sink(), &mut out)?;
        stream.synchronize()?;
        refused.push((boundary, fault::read(gpu.fault_word())?));
        gpu.clear_fault()?;
    }

    let prep = ops.module.prepare_glu(config(1))?;
    for gate in [f32::NEG_INFINITY, f32::INFINITY, f32::NAN] {
        let x = DeviceBuffer::from_host(stream, &[0.0, gate])?;
        ops.module
            .glu(stream, &prep, &x, 1, 1, ops.sink(), &mut out)?;
        stream.synchronize()?;
        refused.push(("GLU gate", fault::read(gpu.fault_word())?));
        gpu.clear_fault()?;
    }

    let q = DeviceBuffer::from_host(stream, &vec![1e20; WIDTH])?;
    let k = DeviceBuffer::from_host(stream, &vec![1e20; WIDTH])?;
    let v = DeviceBuffer::zeroed(stream, WIDTH)?;
    let pos = DeviceBuffer::zeroed(stream, RPE * WIDTH)?;
    let gain = DeviceBuffer::from_host(stream, &vec![1.0; WIDTH / HEADS])?;
    let prep = attention.prepare_local_attention(config(HEADS * 32))?;
    attention.local_attention(
        stream,
        &prep,
        &q,
        &k,
        &v,
        &pos,
        &gain,
        &gain,
        1,
        1.0,
        1.0,
        ops.sink(),
        &mut out,
    )?;
    stream.synchronize()?;
    refused.push(("attention score", fault::read(gpu.fault_word())?));
    for (boundary, got) in &refused {
        eprintln!("{boundary}: {got:?}");
    }
    for (boundary, got) in refused {
        assert_eq!(
            got.and_then(|f| f.site()),
            Some(FaultSite::F32Product),
            "{boundary}: arithmetic overflow was not refused"
        );
    }
    gpu.clear_fault()?;
    Ok(())
}
