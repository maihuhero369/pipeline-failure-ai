// model.js — TF.js model wrapper
const ModelAPI = (function(){
  let model = null;

  function createModel(inputDim, lr=0.01){
    const m = tf.sequential();
    m.add(tf.layers.dense({units:32, activation:'relu', inputShape:[inputDim]}));
    m.add(tf.layers.dense({units:16, activation:'relu'}));
    m.add(tf.layers.dense({units:1, activation:'sigmoid'}));
    m.compile({optimizer: tf.train.adam(lr), loss:'binaryCrossentropy', metrics:['accuracy']});
    model = m;
    return m;
  }

  async function trainModel(X, y, epochs=20, batchSize=32, onEpoch=null){
    if(!model) createModel(X.shape[1]);
    const history = await model.fit(X, y, {
      epochs, batchSize, shuffle:true,
      callbacks: {
        onEpochEnd: async (epoch, logs) => { if(onEpoch) onEpoch(epoch, logs); }
      }
    });
    return history;
  }

  async function predictArray(arr, mins, maxs){
    if(!model) throw new Error('No model loaded');
    const norm = arr.map((v,i)=> (maxs[i]===mins[i])?0.5: (v-mins[i])/(maxs[i]-mins[i]) );
    const out = model.predict(tf.tensor2d([norm]));
    return (await out.data())[0];
  }

  async function saveModelToLocal(){
    if(!model) throw new Error('No model to save');
    await model.save('localstorage://pipeline-failure-model');
  }

  async function loadModelFromLocal(){
    model = await tf.loadLayersModel('localstorage://pipeline-failure-model');
    return model;
  }

  function resetModel(){ model = null; }

  return {createModel, trainModel, predictArray, saveModelToLocal, loadModelFromLocal, resetModel};
})();
