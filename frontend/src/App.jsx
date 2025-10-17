import React from 'react';
import SimulationPanel from './SimulationPanel'; // 新しいコンポーネントをインポート
import './App.css'; // スタイルファイルの存在を確認

function App() {
  return (
    <div className="App">
      {/* SimulationPanel コンポーネントを配置 */}
      <SimulationPanel />
    </div>
  );
}

export default App;