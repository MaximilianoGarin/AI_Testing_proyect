import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [input, setInput] = useState('');
  const [response, setResponse] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post('http://localhost:5000/predict', { data: input });
      setResponse(res.data.prediction);
    } catch (error) {
      console.error('Error al llamar a la API:', error);
    }
  };

  return (
    <div className="App">
      <h1>Asistente Inteligente</h1>
      <form onSubmit={handleSubmit}>
        <input 
          type="text" 
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
        <button type="submit">Enviar</button>
      </form>
      <div>
        <h2>Respuesta:</h2>
        <p>{response}</p>
      </div>
    </div>
  );
}

export default App;