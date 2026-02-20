/**
 * PokedexShell.jsx
 * The outer red plastic hardware body of the Pokédex.
 * Faithfully recreates the reference image:
 *   – Blue lens (top-left) + indicator dots
 *   – Bevelled screen with dot-matrix scanlines
 *   – Red A-button (scan trigger)
 *   – D-pad (decorative)
 *   – Speaker grill lines
 *   – Green data bar
 */

import { useRef } from 'react'
import { AnimatePresence } from 'framer-motion'
import CameraView from './CameraView'
import DexEntry from './DexEntry'
import StatusLED from './StatusLED'

export default function PokedexShell({ phase, result, isScanning, onScan, onScanAgain }) {
    const cameraRef = useRef(null)

    const handleScanClick = async () => {
        if (!cameraRef.current) return
        const blob = await cameraRef.current.captureFrame()
        onScan(blob)
    }

    return (
        <div className="dex-shell">

            {/* ══════════════════════════
          TOP PANEL
      ══════════════════════════ */}
            <div className="dex-top-panel">
                {/* Large blue lens */}
                <div className="dex-lens">
                    <div className="dex-lens-inner" />
                    <div className="dex-lens-glare" />
                </div>

                {/* Indicator dot trio */}
                <div className="dex-dots">
                    <div className="dex-dot dex-dot-red" />
                    <div className="dex-dot dex-dot-yellow" />
                    <div className="dex-dot dex-dot-green" />
                </div>
            </div>

            {/* ══════════════════════════
          SCREEN BEZEL
      ══════════════════════════ */}
            <div className="dex-screen-bezel">
                <div className="dex-screen">
                    {/* Small red LEDs above screen */}
                    <div className="dex-screen-dots">
                        <div className="dex-screen-dot" />
                        <div className="dex-screen-dot" />
                    </div>

                    {/* Screen content area */}
                    <div className="dex-screen-content">
                        <AnimatePresence mode="wait">
                            {phase === 'result' && result ? (
                                <DexEntry
                                    key="result"
                                    result={result}
                                    onScanAgain={onScanAgain}
                                />
                            ) : (
                                <CameraView
                                    key="camera"
                                    ref={cameraRef}
                                    isScanning={isScanning}
                                />
                            )}
                        </AnimatePresence>
                    </div>
                </div>
            </div>

            {/* ══════════════════════════
          CONTROL PANEL
      ══════════════════════════ */}
            <div className="dex-controls">

                {/* Left side: A-button (scan trigger) + LED */}
                <div className="dex-controls-left">
                    <button
                        className={`dex-a-btn ${isScanning ? 'scanning' : ''}`}
                        onClick={handleScanClick}
                        disabled={isScanning || phase === 'result'}
                        aria-label="Scan"
                    >
                        {isScanning ? '…' : '●'}
                    </button>
                    <StatusLED isScanning={isScanning} />
                </div>

                {/* Right side: D-pad */}
                <div className="dex-controls-right">
                    <div className="dex-dpad">
                        <div className="dex-dpad-h" />
                        <div className="dex-dpad-v" />
                        <div className="dex-dpad-centre" />
                    </div>
                </div>

            </div>

            {/* ══════════════════════════
          SPEAKER GRILL
      ══════════════════════════ */}
            <div className="dex-speaker-grill">
                {[...Array(4)].map((_, i) => (
                    <div key={i} className="dex-speaker-line" />
                ))}
            </div>

            {/* ══════════════════════════
          GREEN DATA BAR (bottom)
      ══════════════════════════ */}
            <div className="dex-data-bar">
                <span className="dex-data-bar-text">POKÉDEX v1.0</span>
            </div>

        </div>
    )
}
