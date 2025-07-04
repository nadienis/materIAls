// Variables globales
let model;

// Cargar el modelo al iniciar la página
async function loadModel() {
    try {
        const MODEL_URL = 'model.json'; // Ajusta esta ruta
        console.log("Intentando cargar modelo desde:", MODEL_URL);
        
        // Verifica si el archivo existe
        const response = await fetch(MODEL_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        model = await tf.loadGraphModel(MODEL_URL);
        console.log("Modelo cargado:", model);
        alert("Modelo cargado correctamente ✅");
    } catch (err) {
        console.error("Error al cargar el modelo:", err);
        alert(`Error al cargar el modelo: ${err.message}`);
    }
}

// Función de predicción
async function predict() {
    if (!model) {
        alert("El modelo no está cargado. Intenta recargar la página.");
        return;
    }

    // Obtener valores de entrada
    const er1 = parseFloat(document.getElementById('er1').value);
    const er2 = parseFloat(document.getElementById('er2').value);
    const V1 = parseFloat(document.getElementById('V1').value);
    const V2 = parseFloat(document.getElementById('V2').value);

    // Crear tensor de entrada (ajusta según tu modelo)
    const inputTensor = tf.tensor2d([[er1, er2, V1, V2]]);

    // Predecir
    const predictions = await model.predict(inputTensor);
    const [er, qf, tcf] = predictions;  // Ajusta según las salidas de tu modelo

    // Mostrar resultados
    document.getElementById('er-output').querySelector('.value').textContent = er.dataSync()[0].toFixed(4);
    document.getElementById('qf-output').querySelector('.value').textContent = qf.dataSync()[0].toFixed(4);
    document.getElementById('tcf-output').querySelector('.value').textContent = tcf.dataSync()[0].toFixed(4);

    // Liberar memoria
    inputTensor.dispose();
    tf.dispose(predictions);
}

// Eventos
document.getElementById('predict-btn').addEventListener('click', predict);

// Inicialización
loadModel();
