use std::error::Error;
use std::fmt;
use tensorflow::ops;
use tensorflow::DataType;
use tensorflow::Output;
use tensorflow::Scope;
use tensorflow::Session;
use tensorflow::SessionOptions;
use tensorflow::SessionRunArgs;
use tensorflow::Shape;
use tensorflow::Status;
use tensorflow::Tensor;
use tensorflow::Variable;

use crate::snake::SnakeDirection;

pub fn fitness(apple: u32, frame: u32) -> f64 {
    let base: f64 = 2.0;
    let f_frame = frame as f64;
    let f_apple = apple as f64;
    return f_frame + base.powf(f_apple) + 500.0 * f_apple.powf(2.1)
        - 0.25 * f_frame.powf(1.3) * f_apple.powf(1.2);
}

pub struct SnakeNN {
    session: Session,
    input: tensorflow::Operation,
    output: tensorflow::Output,
}

#[derive(Debug)]
struct MyError(String);

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

impl Error for MyError {}

impl SnakeNN {
    pub fn new() -> Result<SnakeNN, Box<dyn Error>> {
        let mut scope = Scope::new_root_scope();
        // Input layer :
        let input: tensorflow::Operation = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape(Shape::from(&[1u64, 32][..]))
            .build(&mut scope.with_op_name("input"))?;
        // Hidden layer.
        let (vars1, layer1) = layer(
            input.clone(),
            32,
            20,
            &|x, scope| Ok(ops::relu(x, scope)?.into()),
            &mut scope,
        )?;
        // Hidden layer.
        let (vars2, layer2) = layer(
            layer1.clone(),
            20,
            12,
            &|x, scope| Ok(ops::relu(x, scope)?.into()),
            &mut scope,
        )?;
        // Output layer.
        let (vars_output, layer_output) = layer(
            layer2.clone(),
            12,
            4,
            &|x, scope| Ok(ops::sigmoid(x, scope)?.into()),
            &mut scope,
        )?;
        let mut variables = Vec::new();
        variables.extend(vars1);
        variables.extend(vars2);
        variables.extend(vars_output);

        // Initialize variables :
        let options = SessionOptions::new();
        let g = scope.graph_mut();
        let session = Session::new(&options, &g)?;
        let mut run_args = SessionRunArgs::new();
        for variable in &variables {
            run_args.add_target(&variable.initializer());
        }
        session.run(&mut run_args)?;

        let result = Self {
            session: session,
            input: input,
            output: layer_output,
        };
        Ok(result)
    }

    pub fn compute_move(&mut self, inputs: &[f32]) -> Result<SnakeDirection, Box<dyn Error>> {
        let mut input_tensor = Tensor::<f32>::new(&[1, 32]);
        let mut run_args = SessionRunArgs::new();
        for (index, input) in inputs.iter().enumerate() {
            input_tensor[index] = *input;
        }
        run_args.add_feed(&self.input, 0, &input_tensor);

        let result_token = run_args.request_fetch(&self.output.operation, self.output.index);

        self.session.run(&mut run_args)?;

        let result_tensor: Tensor<f32> = run_args.fetch::<f32>(result_token)?;
        let intput_up: f32 = result_tensor.get(&[0, 0]);
        let intput_down: f32 = result_tensor.get(&[0, 1]);
        let intput_left: f32 = result_tensor.get(&[0, 2]);
        let intput_right: f32 = result_tensor.get(&[0, 3]);
        let mut index_save: Option<usize> = None;
        let mut save_value: f32 = f32::NEG_INFINITY;
        for (index, value) in [intput_up, intput_down, intput_left, intput_right]
            .iter()
            .enumerate()
        {
            match index_save {
                Some(v) => {
                    if save_value < *value {
                        index_save = Some(index);
                        save_value = *value;
                    }
                }
                _ => {
                    index_save = Some(index);
                    save_value = *value;
                }
            }
        }
        match index_save {
            Some(v) => match v {
                0 => Ok(SnakeDirection::UP),
                1 => Ok(SnakeDirection::DOWN),
                2 => Ok(SnakeDirection::LEFT),
                3 => Ok(SnakeDirection::RIGHT),
                _ => Err(Box::new(MyError("Oops".into()))),
            },
            _ => Err(Box::new(MyError("Oops".into()))),
        }
    }
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
        let snake_nn = SnakeNN::new();
        match snake_nn {
            Ok(mut nn) => {
                let snake_inputs: [f32; 32] = [
                    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                    1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
                ];
                match nn.compute_move(&snake_inputs) {
                    Ok(_v) => {
                        println!("Looks fine.");
                    }
                    _ => {
                        assert!(false);
                    }
                }
            }
            _ => {
                assert!(false);
            }
        }
    }
}
