pub fn fitness(apple: u32, frame: u32) -> f64 {
    let base: f64 = 2.0;
    let f_frame = frame as f64;
    let f_apple = apple as f64;
    return f_frame + base.powf(f_apple) + 500.0 * f_apple.powf(2.1)
        - 0.25 * f_frame.powf(1.3) * f_apple.powf(1.2);
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
}
