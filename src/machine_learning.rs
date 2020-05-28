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
    bias_initial_value: Vec<Tensor<f32>>,
    weight_initial_value: Vec<Tensor<f32>>,
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
    activation: &dyn Fn(Output, &mut Scope) -> Result<Output, Status>,
    scope: &mut Scope,
    weight_initial_value: Tensor<f32>,
    bias_initial_value: Tensor<f32>,
) -> Result<(Weight, Bias, Output), Status> {
    let mut scope = scope.new_sub_scope("layer");
    let scope = &mut scope;
    let w = Variable::builder()
        .const_initial_value(weight_initial_value.clone())
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
        Ok(SnakeNN::new_with_param(
            32,
            generate_random_standard_normal_tensor(1, 20)?,
            generate_random_standard_normal_tensor(1, 12)?,
            generate_random_standard_normal_tensor(1, 4)?,
            generate_random_standard_normal_tensor(32, 20)?,
            generate_random_standard_normal_tensor(20, 12)?,
            generate_random_standard_normal_tensor(12, 4)?,
        )?)
    }

    pub fn new_with_param(
        input_size: u64,
        bias_hidden_1: Tensor<f32>,
        bias_hidden_2: Tensor<f32>,
        bias_output: Tensor<f32>,
        weight_hidden_1: Tensor<f32>,
        weight_hidden_2: Tensor<f32>,
        weight_output: Tensor<f32>,
    ) -> Result<SnakeNN, Box<dyn Error>> {
        let mut scope = Scope::new_root_scope();
        // Input layer :
        let input: Operation = ops::Placeholder::new()
            .dtype(DataType::Float)
            .shape(Shape::from(&[1u64, input_size][..]))
            .build(&mut scope.with_op_name("input"))?;

        let mut scope_1 = scope.new_sub_scope("layer");
        let scope_1 = &mut scope_1;

        // Hidden layer.
        let (weight_layer_1, bias_layer_1, layer1) = layer(
            input.clone(),
            &|x, scope_1| Ok(ops::relu(x, scope_1)?.into()),
            scope_1,
            weight_hidden_1.clone(),
            bias_hidden_1.clone(),
        )?;

        let mut scope_2 = scope.new_sub_scope("layer");
        let scope_2 = &mut scope_2;

        // Hidden layer.
        let (weight_layer_2, bias_layer_2, layer2) = layer(
            layer1.clone(),
            &|x, scope| Ok(ops::relu(x, scope)?.into()),
            scope_2,
            weight_hidden_2.clone(),
            bias_hidden_2.clone(),
        )?;

        let mut scope_o = scope.new_sub_scope("layer");
        let scope_o = &mut scope_o;

        // Output layer.
        let (weight_layer_out, bias_layer_out, layer_output) = layer(
            layer2.clone(),
            &|x, scope| Ok(ops::sigmoid(x, scope)?.into()),
            scope_o,
            weight_output.clone(),
            bias_output.clone(),
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
            bias_initial_value: vec![bias_hidden_1, bias_hidden_2, bias_output],
            weight_initial_value: vec![weight_hidden_1, weight_hidden_2, weight_output],
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

    fn next(&self) -> Result<SnakeNN, Box<dyn Error>> {
        // TODO : try to avoiding two copy here...
        let bias_1 = self.bias_initial_value[0].clone();
        let bias_2 = self.bias_initial_value[1].clone();
        let bias_o = self.bias_initial_value[2].clone();
        let weight_1 = self.weight_initial_value[0].clone();
        let weight_2 = self.weight_initial_value[1].clone();
        let weight_o = self.weight_initial_value[2].clone();
        mute_gen(&bias_1);
        mute_gen(&bias_2);
        mute_gen(&bias_o);
        mute_gen(&weight_1);
        mute_gen(&weight_2);
        mute_gen(&weight_o);
        SnakeNN::new_with_param(32, bias_1, bias_2, bias_o, weight_1, weight_2, weight_o)
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

fn generate_random_standard_normal_tensor(
    input_size: u64,
    output_size: u64,
) -> Result<Tensor<f32>, Box<dyn Error>> {
    let mut scope = Scope::new_root_scope();
    let mut scope_1 = scope.new_sub_scope("layer");
    let scope_1 = &mut scope_1;

    let w_shape_1 = ops::constant(&[input_size as i64, output_size as i64][..], scope_1)?;
    let w_initial_value_1: Operation = ops::RandomStandardNormal::new()
        .dtype(DataType::Float)
        .build(w_shape_1.into(), scope_1)?;

    let options = SessionOptions::new();
    let g = scope.graph_mut();
    let session = Session::new(&options, &g)?;
    let mut run_args = SessionRunArgs::new();

    let result_token = run_args.request_fetch(&w_initial_value_1, 0);
    session.run(&mut run_args)?;

    let result_tensor: Tensor<f32> = run_args.fetch::<f32>(result_token)?;
    Ok(result_tensor)
}

fn generate_random_uniform_tensor(
    input_size: u64,
    output_size: u64,
) -> Result<Tensor<f32>, Box<dyn Error>> {
    let mut scope = Scope::new_root_scope();
    let mut scope_1 = scope.new_sub_scope("layer");
    let scope_1 = &mut scope_1;

    let w_shape_1 = ops::constant(&[input_size as i64, output_size as i64][..], scope_1)?;
    let w_initial_value_1: Operation = ops::RandomUniform::new()
        .dtype(DataType::Float)
        .build(w_shape_1.into(), scope_1)?;

    let options = SessionOptions::new();
    let g = scope.graph_mut();
    let session = Session::new(&options, &g)?;
    let mut run_args = SessionRunArgs::new();

    let result_token = run_args.request_fetch(&w_initial_value_1, 0);
    session.run(&mut run_args)?;

    let result_tensor: Tensor<f32> = run_args.fetch::<f32>(result_token)?;
    Ok(result_tensor)
}

fn mute_gen(tensor: &Tensor<f32>) -> &Tensor<f32> {
    // TODO : implement it.
    tensor
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

    #[test]
    fn test_generate_weight() -> Result<(), Box<dyn Error>> {
        let weight: Tensor<f32> = generate_random_standard_normal_tensor(32, 20)?;
        let expected_weight = Shape::from(&[32u64, 20][..]);
        assert_eq!(weight.shape(), expected_weight);
        Ok(())
    }
}
