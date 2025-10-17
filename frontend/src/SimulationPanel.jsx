import React, { useState } from 'react';
import axios from 'axios';
import { v4 as uuidv4 } from 'uuid'; // 一意のIDを生成するための例。実際には省略可

// API 設定
// const API_URL = 'http://127.0.0.1:8000/predict_anomaly';
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
        setLastData(dataToSend.time_series); // 送信データを表示用に保存

        try {
            // Axios で FastAPI の POST エンドポイントを呼び出し
            const response = await axios.post(`${API_BASE_URL}${API_ENDPOINT}`, dataToSend);
            
            // バックエンドのJSON応答を受信
            const result = response.data;

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
                    <h4>直近の送信データプレビュー:</h4>
                    <pre style={{ backgroundColor: '#f4f4f4', padding: '10px', borderRadius: '5px' }}>
                        {JSON.stringify({ time_series: lastData }, null, 2)}
                    </pre>
                </div>
            )}
        </div>
    );
};

export default SimulationPanel;