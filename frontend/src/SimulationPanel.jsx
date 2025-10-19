import React, { useState } from 'react';
import axios from 'axios';

// API 設定
const API_BASE_URL = 'http://127.0.0.1:8000'; // ベースURL
const API_ENDPOINT = '/predict_anomaly';

// --- シミュレーションデータ生成関数 ---
const generateSimulationData = (isAnomaly) => {
    const time_series = [];
    const now = new Date();
    
    // 直近10分の3データポイントを模擬
    for (let i = 2; i >= 0; i--) {
        const timestamp = new Date(now.getTime() - i * 10 * 60000); // 10分ごとに1点
        let power_kW;

        if (isAnomaly) {
            // 異常データ: 夜間や非ピーク時の突発的な高消費電力を模擬
            power_kW = Math.random() * 3.0 + 4.0; // 4.0〜7.0 kW/h
        } else {
            // 正常データ: 通常の低消費電力を模擬（例: 夜間 0.2〜0.5 kW/h）
            power_kW = Math.random() * 0.3 + 0.2; // 0.2〜0.5 kW/h
        }

        time_series.push({
            // 注意: FastAPI は ISO 8601 形式の文字列タイムスタンプを要求
            timestamp: timestamp.toISOString(), 
            power_kW: parseFloat(power_kW.toFixed(2)),
        });
    }

    return { time_series };
};


// --- SimulationPanel コンポーネント ---
const SimulationPanel = () => {
    // 状態管理
    const [status, setStatus] = useState('実行待ち...');
    const [score, setScore] = useState(null);
    const [message, setMessage] = useState('正常または異常のデータストリームを送信してください。');
    const [loading, setLoading] = useState(false);
    const [lastData, setLastData] = useState([]);

    // 状態に応じたスタイル設定
    const statusStyle = {
        padding: '10px',
        borderRadius: '5px',
        fontWeight: 'bold',
        color: status === 'Warning' ? 'white' : 'black',
        backgroundColor: 
            status === 'Warning' ? '#dc3545' : 
            status === 'Normal' ? '#28a745' : '#6c757d',
    };

    // データ送信の中核関数
    const sendData = async (isAnomaly) => {
        setLoading(true);
        setStatus('データ送信中、AI の予測を待機しています...');
        setMessage('');

        const dataToSend = generateSimulationData(isAnomaly);

        try {
            // Axios で FastAPI の POST エンドポイントを呼び出し
            const response = await axios.post(`${API_BASE_URL}${API_ENDPOINT}`, dataToSend);
            
            // バックエンドのJSON応答を受信
            const result = response.data;

            // ✅ 変更点：API 応答から完全なスコア付きデータリストを取得
            if (result && result.submitted_data_preview && Array.isArray(result.submitted_data_preview)) {
                // 状態変数は setLastData に割り当て
                setLastData(result.submitted_data_preview);
            } else {
                // データ構造が予期しないエラーを処理
                console.error("API 応答データ構造エラー: submitted_data_preview キーがありません");
                setLastData([]); // エラー時にリストを空にする
            }
            
            setStatus(result.status);
            setScore(result.average_anomaly_score.toFixed(4));
            setMessage(result.message);

        } catch (error) {
            console.error("API 呼び出し失敗:", error);
            setStatus('Error');
            setMessage('API 呼び出しに失敗しました。バックエンドが起動しているか（ポート8000）、CORS エラーがないか確認してください。');
            setScore(null);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto', fontFamily: 'Arial, sans-serif' }}>
            <h2>AIoT デバイス稼働状態シミュレーター</h2>
            <p>下のボタンをクリックして、バックエンドAIサービスへ2種類の電力データストリームを送信します:</p>

            {/* --- 1. 入力エリア: ボタン --- */}
            <div style={{ marginBottom: '20px' }}>
                <button 
                    onClick={() => sendData(false)} 
                    disabled={loading}
                    style={{ 
                        marginRight: '10px', 
                        padding: '10px 20px', 
                        backgroundColor: '#28a745', 
                        color: 'white', 
                        border: 'none', 
                        cursor: loading ? 'not-allowed' : 'pointer' 
                    }}
                >
                    {loading && lastData.length === 0 ? '読み込み中...' : '送信 正常データ（低消費電力）'}
                </button>
                <button 
                    onClick={() => sendData(true)} 
                    disabled={loading}
                    style={{ 
                        padding: '10px 20px', 
                        backgroundColor: '#dc3545', 
                        color: 'white', 
                        border: 'none', 
                        cursor: loading ? 'not-allowed' : 'pointer' 
                    }}
                >
                    {loading && lastData.length === 0 ? '読み込み中...' : '送信 異常データ（高消費電力）'}
                </button>
            </div>

            {/* --- 2. 表示エリア: ステータスとスコア --- */}
            <div style={{ border: '1px solid #ccc', padding: '15px', borderRadius: '8px' }}>
                <h3>AI 予測結果</h3>
                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '10px' }}>
                    <p style={{ marginRight: '20px' }}>**現在の状態:**</p>
                    <div style={statusStyle}>
                        {status}
                    </div>
                </div>
                
                {score !== null && (
                    <p>**平均異常スコア:** <span style={{ color: score < 0.1 ? 'red' : 'green' }}>{score}</span></p>
                )}
                
                <p>**AI 説明:** {message}</p>
            </div>

            {/* --- データ送信プレビュー --- */}
            {lastData.length > 0 && (
                <div style={{ marginTop: '20px', fontSize: '0.9em' }}>
                    <h4>直近の送信データプレビュー (AI 予測結果を含む):</h4>
                    
                    {/* データを表やリストで表示することで、JSON.stringifyより分かりやすくなります */}
                    <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
                        <thead>
                            <tr style={{ backgroundColor: 'transparent' }}>
                                <th style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>タイムスタンプ (UTC)</th>
                                <th style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>電力 (kW)</th>
                                <th style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>異常スコア</th>
                                <th style={{ padding: '8px', borderBottom: '1px solid #ddd' }}>予測</th>
                            </tr>
                        </thead>
                        <tbody>
                            {/* lastDataを繰り返し処理し、現在は anomaly_score と is_anomaly を含みます */}
                            {lastData.map((dataPoint, index) => (
                                <tr key={index} style={{ backgroundColor: dataPoint.is_anomaly === -1 ? 'transparent' : 'inherit' }}>
                                    <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>
                                        {new Date(dataPoint.timestamp).toLocaleString('ja-JP', { 
                                            year: 'numeric', 
                                            month: '2-digit', 
                                            day: '2-digit', 
                                            hour: '2-digit', 
                                            minute: '2-digit', 
                                            second: '2-digit',
                                            hour12: false // 24 時間制
                                        })}
                                    </td>
                                    <td style={{ padding: '8px', borderBottom: '1px solid #eee' }}>
                                        {dataPoint.power_kW.toFixed(2)}
                                    </td>
                                    <td style={{ padding: '8px', borderBottom: '1px solid #eee', color: dataPoint.anomaly_score < 0.05 ? 'red' : 'green' }}>
                                        {dataPoint.anomaly_score.toFixed(4)}
                                    </td>
                                    <td style={{ padding: '8px', borderBottom: '1px solid #eee', fontWeight: 'bold' }}>
                                        {dataPoint.is_anomaly === -1 ? '⚠️ 異常' : '✅ 正常'}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
};

export default SimulationPanel;