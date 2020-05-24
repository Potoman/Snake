use std::{error::Error, fmt};
use tensorflow::{
    ops, DataType, Operation, Output, Scope, Session, SessionOptions, SessionRunArgs, Shape,
    Status, Tensor, Variable,
};

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
    input: Operation,
    output: Output,
    weight_layer_1: Weight,
    weight_layer_2: Weight,
    weight_layer_out: Weight,
    bias_layer_1: Bias,
    bias_layer_2: Bias,
    bias_layer_out: Bias,
    bias_initial_value: Vec<Tensor<f32>>,
    weight_initial_value: Vec<Operation>,
}

#[derive(Debug)]
struct MyError(String);

impl fmt::Display for MyError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "There is an error: {}", self.0)
    }
}

impl Error for MyError {}

struct Bias {
    variable: Variable,
}

struct Weight {
    variable: Variable,
}

fn layer<O1: Into<Output>>(
    input: O1,
    input_size: u64,
    output_size: u64,
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
    weight_initial_value: Operation,
    bias_initial_value: Tensor<f32>,
) -> Result<(Weight, Bias, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w = Variable::builder()
        .initial_value(weight_initial_value.clone())
        .data_type(DataType::Float)
        .shape(Shape::from(&[input_size, output_size][..]))
        .build(&mut scope.with_op_name("w"))?;
    let b = Variable::builder()
        .const_initial_value(bias_initial_value.clone())
        .build(&mut scope.with_op_name("b"))?;
    Ok((
        Weight {
            variable: w.clone(),
        },
        Bias {
            variable: b.clone(),
        },
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

impl SnakeNN {
    pub fn new() -> Result<SnakeNN, Box<dyn Error>> {
        let mut scope = Scope::new_root_scope();
        // Input layer :
        let input: Operation = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape(Shape::from(&[1u64, 32][..]))
            .build(&mut scope.with_op_name("input"))?;

        let mut scope_1 = scope.new_sub_scope("layer");
        let scope_1 = &mut scope_1;

        let w_shape_1 = ops::constant(&[32 as i64, 20 as i64][..], scope_1)?;
        let w_initial_value_1: Operation = ops::RandomStandardNormal::new()
            .dtype(DataType::Float)
            .build(w_shape_1.into(), scope_1)?;
        let b_initial_value_1: Tensor<f32> = Tensor::<f32>::new(&[20]);

        // Hidden layer.
        let (weight_layer_1, bias_layer_1, layer1) = layer(
            input.clone(),
            32,
            20,
            &|x, scope_1| Ok(ops::relu(x, scope_1)?.into()),
            scope_1,
            w_initial_value_1.clone(),
            b_initial_value_1.clone(),
        )?;

        let mut scope_2 = scope.new_sub_scope("layer");
        let scope_2 = &mut scope_2;

        let w_shape_2 = ops::constant(&[20 as i64, 12 as i64][..], scope_2)?;
        let w_initial_value_2: Operation = ops::RandomStandardNormal::new()
            .dtype(DataType::Float)
            .build(w_shape_2.into(), scope_2)?;
        let b_initial_value_2: Tensor<f32> = Tensor::<f32>::new(&[12]);

        // Hidden layer.
        let (weight_layer_2, bias_layer_2, layer2) = layer(
            layer1.clone(),
            20,
            12,
            &|x, scope| Ok(ops::relu(x, scope)?.into()),
            scope_2,
            w_initial_value_2.clone(),
            b_initial_value_2.clone(),
        )?;

        let mut scope_o = scope.new_sub_scope("layer");
        let scope_o = &mut scope_o;

        let w_shape_o = ops::constant(&[12 as i64, 4 as i64][..], scope_o)?;
        let w_initial_value_o: Operation = ops::RandomStandardNormal::new()
            .dtype(DataType::Float)
            .build(w_shape_o.into(), scope_o)?;
        let b_initial_value_o: Tensor<f32> = Tensor::<f32>::new(&[4]);

        // Output layer.
        let (weight_layer_out, bias_layer_out, layer_output) = layer(
            layer2.clone(),
            12,
            4,
            &|x, scope| Ok(ops::sigmoid(x, scope)?.into()),
            scope_o,
            w_initial_value_o.clone(),
            b_initial_value_o.clone(),
        )?;

        // Initialize variables :
        let options = SessionOptions::new();
        let g = scope.graph_mut();
        let session = Session::new(&options, &g)?;
        let mut run_args = SessionRunArgs::new();
        run_args.add_target(&weight_layer_1.variable.initializer());
        run_args.add_target(&weight_layer_2.variable.initializer());
        run_args.add_target(&weight_layer_out.variable.initializer());
        run_args.add_target(&bias_layer_1.variable.initializer());
        run_args.add_target(&bias_layer_2.variable.initializer());
        run_args.add_target(&bias_layer_out.variable.initializer());

        session.run(&mut run_args)?;

        let result = Self {
            session: session,
            input: input,
            output: layer_output,
            weight_layer_1,
            weight_layer_2,
            weight_layer_out,
            bias_layer_1,
            bias_layer_2,
            bias_layer_out,
            bias_initial_value: vec![b_initial_value_1, b_initial_value_2, b_initial_value_o],
            weight_initial_value: vec![w_initial_value_1, w_initial_value_2, w_initial_value_o],
        };
        Ok(result)
    }

    fn compute_nn_output(&mut self, inputs: &[f32]) -> Result<[f32; 4], Box<dyn Error>> {
        let mut input_tensor = Tensor::<f32>::new(&[1, 32]);
        for (index, input) in inputs.iter().enumerate() {
            input_tensor[index] = *input;
        }
        let mut run_args = SessionRunArgs::new();
        run_args.add_feed(&self.input, 0, &input_tensor);

        let result_token = run_args.request_fetch(&self.output.operation, self.output.index);

        self.session.run(&mut run_args)?;

        let result_tensor: Tensor<f32> = run_args.fetch::<f32>(result_token)?;
        Ok([
            result_tensor.get(&[0, 0]),
            result_tensor.get(&[0, 1]),
            result_tensor.get(&[0, 2]),
            result_tensor.get(&[0, 3]),
        ])
    }

    fn next() -> Result<SnakeNN, Box<dyn Error>> {
        // Here will be a new NN in order to compute the next direction for the snake.
        // It will contains the crossover and the mutation.
        SnakeNN::new()
    }
}

fn compute_move(inputs: &[f32; 4]) -> SnakeDirection {
    let mut index_save: Option<usize> = None;
    let mut save_value: f32 = f32::NEG_INFINITY;
    for (index, value) in inputs.iter().enumerate() {
        match index_save {
            Some(_v) => {
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
            0 => SnakeDirection::UP,
            1 => SnakeDirection::DOWN,
            2 => SnakeDirection::LEFT,
            3 => SnakeDirection::RIGHT,
            _ => SnakeDirection::UP,
        },
        _ => SnakeDirection::UP,
    }
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
    fn test_compute_nn_output() -> Result<(), Box<dyn Error>> {
        let mut nn = SnakeNN::new()?;
        let snake_inputs: [f32; 32] = [
            1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0,
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let output_0 = nn.compute_nn_output(&snake_inputs)?;
        let output_1 = nn.compute_nn_output(&snake_inputs)?;
        assert_eq!(output_0, output_1);
        Ok(())
    }

    #[test]
    fn test_compute_move() -> Result<(), Box<dyn Error>> {
        {
            let snake_direction = compute_move(&[1.0, 0.0, 0.0, 0.0]);
            assert_eq!(snake_direction, SnakeDirection::UP);
        }
        {
            let snake_direction = compute_move(&[1.0, 1.0, 0.0, 0.0]);
            assert_eq!(snake_direction, SnakeDirection::UP);
        }
        {
            let snake_direction = compute_move(&[1.0, 1.0, 1.0, 0.0]);
            assert_eq!(snake_direction, SnakeDirection::UP);
        }
        {
            let snake_direction = compute_move(&[1.0, 1.0, 1.0, 1.0]);
            assert_eq!(snake_direction, SnakeDirection::UP);
        }
        {
            let snake_direction = compute_move(&[0.0, 1.0, 0.0, 0.0]);
            assert_eq!(snake_direction, SnakeDirection::DOWN);
        }
        {
            let snake_direction = compute_move(&[0.0, 0.0, 1.0, 0.0]);
            assert_eq!(snake_direction, SnakeDirection::LEFT);
        }
        {
            let snake_direction = compute_move(&[0.0, 0.0, 0.0, 1.0]);
            assert_eq!(snake_direction, SnakeDirection::RIGHT);
        }
        Ok(())
    }
}
