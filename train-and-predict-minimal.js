const tf = require('@tensorflow/tfjs');
const table = require('table').table;

// Minimal train and predict demo
async function run() {
  // Create a sequential model (outputs of one layer are the inputs to the next layer)
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training, specifying the loss and optimizer functions
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Create synthetic training data (y = 2x - 1)
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);

  // Train the model
  await model.fit(xs, ys, {epochs: 250});

  // Predict outputs for a set of test inputs
  const testInputs = [3, 14, 25, 43, 56, 72];
  const testOutputs = testInputs.map(input =>
    model.predict(tf.tensor2d([input], [1, 1]))
  );

  // Display training data in tabular format
  var trainingDataTabular = [['x', 'y']];
  for (var i = 0; i < Math.min(xs.size, ys.size); i++) {
    trainingDataTabular.push([xs.get(i, 0), ys.get(i, 0)]);
  }
  console.log('========================================');
  console.log('Training Data');
  console.log('========================================');
  console.log(table(trainingDataTabular));
  // Display prediction results in tabular format
  var testInferencingResultsTabular = [['x', 'y inferenced', 'y calculated']];
  for (var i = 0; i < Math.min(testInputs.length, testOutputs.length); i++) {
    testInferencingResultsTabular.push([
      testInputs[i],
      testOutputs[i].dataSync(),
      2 * testInputs[i] - 1
    ]);
  }
  console.log('========================================');
  console.log('Prediction Results on Test Data');
  console.log('========================================');
  console.log(table(testInferencingResultsTabular));
}

run();
