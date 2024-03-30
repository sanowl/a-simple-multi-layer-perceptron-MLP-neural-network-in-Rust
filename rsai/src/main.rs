extern create tensorflow;

use tensorflow::{Graph, Session, SessionOptions, Tensor};

fn main() {
    // Create a new TensorFlow graph
    let mut graph = Graph::new();

    // Create two placeholder tensors
    let a = Tensor::new(&[1]).with_values(&[2i32]).unwrap();
    let b = Tensor::new(&[1]).with_values(&[3i32]).unwrap();

    // Add an operation to the graph that adds the two tensors
    let c = graph.new_operation("Add", "c").unwrap();
    c.add_input(a.into());
    c.add_input(b.into());

    // Finalize the graph
    let c = c.finish().unwrap();

    // Create a new TensorFlow session and run the graph
    let mut session = Session::new(&SessionOptions::new(), &graph).unwrap();
    let result = session.run(&[], &[c], &[]).unwrap();

    // Print the result
    println!("{:?}", result[0]);
}
