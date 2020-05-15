use std::error::Error;
use tensorflow::ops;
use tensorflow::DataType;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Shape;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::Variable;

pub fn fitness(apple: u32, frame: u32) -> f64 {
    let base: f64 = 2.0;
    let f_frame = frame as f64;
    let f_apple = apple as f64;
    return f_frame + base.powf(f_apple) + 500.0 * f_apple.powf(2.1)
        - 0.25 * f_frame.powf(1.3) * f_apple.powf(1.2);
}

fn layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
) -> Result<(Vec<Variable>, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w_shape = ops::constant(&[input_size as i64, output_size as i64][..], scope)?;
    let w = Variable::builder()
        .initial_value(
            ops::RandomStandardNormal::new()
                .dtype(DataType::Float)
                .build(w_shape.into(), scope)?,
        )
        .data_type(DataType::Float)
        .shape(Shape::from(&[input_size, output_size][..]))
        .build(&mut scope.with_op_name("w"))?;
    let b = Variable::builder()
        .const_initial_value(Tensor::<f32>::new(&[output_size]))
        .build(&mut scope.with_op_name("b"))?;
    Ok((
        vec![w.clone(), b.clone()],
        activation(
            ops::add(
                ops::mat_mul(input.into(), w.output().clone(), scope)?.into(),
                b.output().clone(),
                scope,
            )?
            .into(),
            scope,
        )?,
    ))
}

pub fn build_model() -> Result<(), Box<dyn Error>> {
    let mut scope = Scope::new_root_scope();
    let scope = &mut scope;
    // Input layer :
    let input = ops::Placeholder::new()
        .dtype(DataType::Float)
        .shape(Shape::from(&[1u64, 32][..]))
        .build(&mut scope.with_op_name("input"))?;
    // Hidden layer.
    let (vars1, layer1) = layer(
        input.clone(),
        32,
        20,
        &|x, scope| Ok(ops::relu(x, scope)?.into()),
        scope,
    )?;
    // Hidden layer.
    let (vars2, layer2) = layer(
        layer1.clone(),
        20,
        12,
        &|x, scope| Ok(ops::relu(x, scope)?.into()),
        scope,
    )?;
    // Output layer.
    let (vars_output, layer_output) = layer(
        layer2.clone(),
        12,
        4,
        &|x, scope| Ok(ops::sigmoid(x, scope)?.into()),
        scope,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fitness() {
        assert_eq!(fitness(0, 0), 1.0);
        assert_eq!(fitness(0, 1), 2.0);
        assert_eq!(fitness(1, 0), 502.0);
        assert_eq!(fitness(1, 1), 502.75);
    }

    #[test]
    fn test_fitness_huge_value() {
        assert_eq!(
            fitness(16 * 16, 16 * 16 * 1000),
            115792089237316200000000000000000000000000000000000000000000000000000000000000.0
        );
    }

    #[test]
    fn test_nn() {
        match build_model() {
            _ => {}
        }
    }
}
