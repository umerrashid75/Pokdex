/**
 * App.jsx – Root component.
 * State machine:
 *   'intro'   → PokeBallIntro animation
 *   'scanner' → PokedexShell with live camera
 *   'result'  → PokedexShell with DexEntry (slide-in result)
 *
 * Layout behaviour:
 *   Mobile  → shell fills 100% viewport edge-to-edge (phone IS the Pokédex)
 *   Desktop → centered shell on dark bg + "use mobile" banner
 */

import { useState, useCallback, useEffect } from 'react'
import PokeBallIntro from './components/PokeBallIntro'
import PokedexShell from './components/PokedexShell'
import { scanImage } from './api/scanApi'

// Anything wider than a phone in portrait
const MOBILE_BREAKPOINT = 768

export default function App() {
    const [phase, setPhase] = useState('intro')
    const [isScanning, setIsScanning] = useState(false)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isDesktop, setIsDesktop] = useState(false)

    // Detect viewport on mount + resize
    useEffect(() => {
        const check = () => setIsDesktop(window.innerWidth >= MOBILE_BREAKPOINT)
        check()
        window.addEventListener('resize', check)
        return () => window.removeEventListener('resize', check)
    }, [])

    const handleIntroComplete = useCallback(() => setPhase('scanner'), [])

    const handleScan = useCallback(async (imageBlob) => {
        setIsScanning(true)
        setError(null)
        try {
            const data = await scanImage(imageBlob)
            setResult(data)
            setPhase('result')
        } catch (err) {
            setError(err?.response?.data?.detail ?? 'Scan failed. Try again.')
        } finally {
            setIsScanning(false)
        }
    }, [])

    const handleScanAgain = useCallback(() => {
        setResult(null)
        setError(null)
        setPhase('scanner')
    }, [])

    return (
        <div className={`app-root ${isDesktop ? 'is-desktop' : 'is-mobile'}`}>

            {/* Desktop "better on mobile" banner */}
            {isDesktop && (
                <div className="desktop-banner">
                    <span className="desktop-banner-icon">📱</span>
                    <span className="desktop-banner-text">
                        For the full Pokédex experience, open this on your phone.
                    </span>
                </div>
            )}

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

            {error && <div className="error-toast">⚠ {error}</div>}
        </div>
    )
}
