#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_predict() {
        let mlp = MLP::new(&[2, 2, 1], 0.1);
        let inputs = vec![1.0, 1.0];
        let output = mlp.predict(&inputs);
        assert_eq!(output, 0.0);
    }

    #[test]
    fn test_mlp_train() {
        let mut mlp = MLP::new(&[2, 2, 1], 0.1);
        let inputs = vec![1.0, 1.0];
        let true_label = 1.0;
        let loss = mlp.train(&inputs, true_label);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_mlp_evaluate() {
        let mlp = MLP::new(&[2, 2, 1], 0.1);
        let test_data = vec![
            (vec![1.0, 1.0], 1.0),
            (vec![1.0, -1.0], -1.0),
            (vec![-1.0, 1.0], -1.0),
            (vec![-1.0, -1.0], -1.0),
        ];
        let accuracy = mlp.evaluate(&test_data);
        assert_eq!(accuracy, 1.0);
    }
}