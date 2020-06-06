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
            TensorProvider.gen([1, 20])?,
            TensorProvider.gen([1, 12])?,
            TensorProvider.gen([1, 4])?,
            TensorProvider.gen([32, 20])?,
            TensorProvider.gen([20, 12])?,
            TensorProvider.gen([12, 4])?,
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
        let mut bias_1 = self.bias_initial_value[0].clone();
        let mut bias_2 = self.bias_initial_value[1].clone();
        let mut bias_o = self.bias_initial_value[2].clone();
        let mut weight_1 = self.weight_initial_value[0].clone();
        let mut weight_2 = self.weight_initial_value[1].clone();
        let mut weight_o = self.weight_initial_value[2].clone();
        SnakeNN::new_with_param(
            32,
            mute_gen(&mut bias_1)?,
            mute_gen(&mut bias_2)?,
            mute_gen(&mut bias_o)?,
            mute_gen(&mut weight_1)?,
            mute_gen(&mut weight_2)?,
            mute_gen(&mut weight_o)?,
        )
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

trait OverloadedGenTensor<T> {
    fn overloaded_gen_tensor(&self, size: T) -> Result<Tensor<f32>, Box<dyn Error>>;
}

struct TensorProvider;

impl TensorProvider {
    fn gen<T>(&self, size: T) -> Result<Tensor<f32>, Box<dyn Error>>
    where
        Self: OverloadedGenTensor<T>,
    {
        self.overloaded_gen_tensor(size)
    }
}

impl OverloadedGenTensor<[u64; 2]> for TensorProvider {
    fn overloaded_gen_tensor(&self, size: [u64; 2]) -> Result<Tensor<f32>, Box<dyn Error>> {
        let mut scope = Scope::new_root_scope();
        let mut scope_1 = scope.new_sub_scope("layer");
        let scope_1 = &mut scope_1;

        let w_shape_1 = ops::constant(&[size[0], size[1]][..], scope_1)?;
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
}

impl OverloadedGenTensor<Shape> for TensorProvider {
    fn overloaded_gen_tensor(&self, size: Shape) -> Result<Tensor<f32>, Box<dyn Error>> {
        let a = size[0];
        match a {
            Some(v) => {
                let b = size[1];
                match b {
                    Some(w) => self.overloaded_gen_tensor([v as u64, w as u64]),
                    _ => Err(Box::new(MyError("No dimension for Shape.".into()))),
                }
            }
            _ => Err(Box::new(MyError("No dimension for Shape.".into()))),
        }
    }
}

fn generate_random_standard_normal_tensor(size: [i64; 2]) -> Result<Tensor<f32>, Box<dyn Error>> {
    let mut scope = Scope::new_root_scope();
    let mut scope_1 = scope.new_sub_scope("layer");
    let scope_1 = &mut scope_1;

    let w_shape_1 = ops::constant(&[size[0], size[1]][..], scope_1)?;
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

struct Adder(Tensor<f32>);

impl std::ops::Add<&Tensor<f32>> for Adder {
    type Output = Tensor<f32>;

    fn add(mut self, _tensor: &Tensor<f32>) -> Tensor<f32> {
        for i in _tensor.shape()[0] {
            for j in _tensor.shape()[1] {
                let x = i as u64;
                let y = j as u64;
                self.0
                    .set(&[x, y], self.0.get(&[x, y]) + _tensor.get(&[x, y]));
            }
        }
        self.0
    }
}

fn mute_gen(tensor: &Tensor<f32>) -> Result<Tensor<f32>, Box<dyn Error>> {
    let output_tensor = TensorProvider.gen(tensor.shape())?;
    Ok(Adder(output_tensor) + tensor)
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
    fn test_generate_random_standard_normal_tensor() -> Result<(), Box<dyn Error>> {
        let weight: Tensor<f32> = generate_random_standard_normal_tensor([32, 20])?;
        let expected_weight = Shape::from(&[32u64, 20][..]);
        assert_eq!(weight.shape(), expected_weight);
        Ok(())
    }

    #[test]
    fn test_generate_random_uniform_tensor() -> Result<(), Box<dyn Error>> {
        let weight: Tensor<f32> = TensorProvider.gen([32, 20])?;
        let expected_weight = Shape::from(&[32u64, 20][..]);
        assert_eq!(weight.shape(), expected_weight);
        Ok(())
    }
}
