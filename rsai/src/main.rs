extern crate plotters;

use plotters::prelude::*;
use std::f64;

struct Layer {
    weights: Vec<f64>,
    bias: f64,
}

impl Layer {
    fn new(input_dim: usize) -> Self {
        let weights = vec![0.0; input_dim];
        let bias = 0.0;
        Layer { weights, bias }
    }

    fn forward(&self, inputs: &Vec<f64>) -> f64 {
        let sum: f64 = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        sum + self.bias
    }

    fn backward(&mut self, inputs: &Vec<f64>, error: f64, learning_rate: f64) {
        for (w, i) in self.weights.iter_mut().zip(inputs) {
            *w += learning_rate * error * i;
        }
        self.bias += learning_rate * error;
    }
}

struct MLP {
    layers: Vec<Layer>,
    learning_rate: f64,
}

/// Represents a Multi-Layer Perceptron (MLP) neural network.
struct MLP {
    layers: Vec<Layer>,  // The layers of the MLP
    learning_rate: f64,  // The learning rate for training the MLP
}

impl MLP {
    /// Creates a new instance of the MLP with the given layer sizes and learning rate.
    ///
    /// # Arguments
    ///
    /// * `layer_sizes` - A slice of usize values representing the sizes of each layer in the MLP.
    /// * `learning_rate` - The learning rate for training the MLP.
    ///
    /// # Returns
    ///
    /// A new instance of the MLP.
    fn new(layer_sizes: &[usize], learning_rate: f64) -> Self {
        let layers: Vec<Layer> = layer_sizes.windows(2).map(|sizes| Layer::new(sizes[0])).collect();
        MLP { layers, learning_rate }
    }

    /// Predicts the output of the MLP for the given inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of f64 values representing the input to the MLP.
    ///
    /// # Returns
    ///
    /// The predicted output of the MLP.
    fn predict(&self, inputs: &Vec<f64>) -> f64 {
        let mut layer_inputs = inputs.clone();
        for layer in &self.layers {
            layer_inputs = vec![layer.forward(&layer_inputs)];
        }
        layer_inputs[0]
    }

    /// Trains the MLP using the given inputs and true label.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A vector of f64 values representing the input to the MLP.
    /// * `true_label` - The true label corresponding to the inputs.
    ///
    /// # Returns
    ///
    /// The squared error as the loss during training.
    fn train(&mut self, inputs: &Vec<f64>, true_label: f64) -> f64 {
        let mut layer_inputs = vec![inputs.clone()];
        let mut layer_outputs = vec![inputs.clone()];
        for layer in &self.layers {
            let output = layer.forward(layer_inputs.last().unwrap());
            layer_inputs.push(vec![output]);
            layer_outputs.push(vec![if output >= 0.0 { 1.0 } else { -1.0 }]);
        }
        let mut error = true_label - *layer_outputs.last().unwrap().last().unwrap();
        for (layer, inputs) in self.layers.iter_mut().zip(layer_inputs.iter()).rev() {
            layer.backward(inputs, error, self.learning_rate);
            error = layer.weights[0] * error;
        }
        error.powi(2)  // return the squared error as the loss
    }

    /// Evaluates the performance of the MLP on the given test data.
    ///
    /// # Arguments
    ///
    /// * `test_data` - A slice of tuples containing the test inputs and true labels.
    ///
    /// # Returns
    ///
    /// The accuracy of the MLP on the test data, represented as a value between 0.0 and 1.0.
    fn evaluate(&self, test_data: &[(Vec<f64>, f64)]) -> f64 {
        let mut correct_predictions = 0;
        for &(ref inputs, true_label) in test_data {
            if (self.predict(inputs) >= 0.0) == (true_label >= 0.0) {
                correct_predictions += 1;
            }
        }
        correct_predictions as f64 / test_data.len() as f64
    }
}

fn main() {
    let mut mlp = MLP::new(&[2, 2, 1], 0.1);
    let training_data = vec![
        (vec![1.0, 1.0], 1.0),
        (vec![1.0, -1.0], -1.0),
        (vec![-1.0, 1.0], -1.0),
        (vec![-1.0, -1.0], -1.0),
    ];
    let mut losses = Vec::new();
    for _ in 0..100 {
        let mut epoch_loss = 0.0;
        for &(ref inputs, true_label) in &training_data {
            epoch_loss += mlp.train(inputs, true_label);
        }
        losses.push(epoch_loss / training_data.len() as f64);
    }
    let accuracy = mlp.evaluate(&training_data);
    println!("Accuracy: {}", accuracy);
    
    // Plot the loss
    let root = BitMapBackend::new("loss.png", (640, 480)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let mut chart = ChartBuilder::on(&root)
        .caption("Training Loss", ("sans-serif", 20).into_font())
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0..100, 0f64..*losses.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap())
        .unwrap();
    chart.configure_mesh().draw().unwrap();
    chart.draw_series(LineSeries::new((0..100).zip(losses.iter().cloned()), &RED)).unwrap();  // modified line
    root.present().unwrap();
}