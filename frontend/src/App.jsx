/**
 * App.jsx – Root component.
 * State machine:
 *   'intro'   → PokeBallIntro animation
 *   'scanner' → PokedexShell with live camera
 *   'result'  → PokedexShell with DexEntry (slide-in result)
 */

import { useState, useCallback } from 'react'
import PokeBallIntro from './components/PokeBallIntro'
import PokedexShell from './components/PokedexShell'
import { scanImage } from './api/scanApi'

export default function App() {
    const [phase, setPhase] = useState('intro')   // 'intro' | 'scanner' | 'result'
    const [isScanning, setIsScanning] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)

    // Called when the PokeBall intro finishes
    const handleIntroComplete = useCallback(() => {
        setPhase('scanner')
    }, [])

    // Called when user presses the red A-button
    const handleScan = useCallback(async (imageBlob) => {
        setIsScanning(true)
        setError(null)

        try {
            const data = await scanImage(imageBlob)
            setResult(data)
            setPhase('result')
        } catch (err) {
            console.error('Scan failed:', err)
            setError(err?.response?.data?.detail ?? 'Scan failed. Try again.')
        } finally {
            setIsScanning(false)
        }
    }, [])

    // Called when user taps "Scan Again"
    const handleScanAgain = useCallback(() => {
        setResult(null)
        setError(null)
        setPhase('scanner')
    }, [])

    return (
        <div className="app-root">
            {phase === 'intro' && (
                <PokeBallIntro onComplete={handleIntroComplete} />
            )}

            {(phase === 'scanner' || phase === 'result') && (
                <PokedexShell
                    phase={phase}
                    result={result}
                    isScanning={isScanning}
                    onScan={handleScan}
                    onScanAgain={handleScanAgain}
                />
            )}

            {/* Global error toast */}
            {error && (
                <div className="error-toast">
                    ⚠ {error}
                </div>
            )}
        </div>
    )
}
