/**
 * DexEntry.jsx
 * Displays the Pokedex scan result with:
 *  - Species name + type badge
 *  - Confidence percentage bar
 *  - Typewriter-animated "Dex Entry" lore text
 *  - A "Scan Again" button to return to camera view
 */

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'

// Map type strings to accent colours matching the reference image palette
const TYPE_COLORS = {
    Mammal: '#E8A000',
    Bird: '#4FC3F7',
    Fish: '#0288D1',
    Reptile: '#66BB6A',
    Unknown: '#9E9E9E',
}

export default function DexEntry({ result, onScanAgain }) {
    const { display_name, type, confidence, dex_entry } = result

    const [displayedText, setDisplayedText] = useState('')
    const indexRef = useRef(0)
    const timerRef = useRef(null)

    // Typewriter effect
    useEffect(() => {
        setDisplayedText('')
        indexRef.current = 0

        timerRef.current = setInterval(() => {
            indexRef.current += 1
            setDisplayedText(dex_entry.slice(0, indexRef.current))

            if (indexRef.current >= dex_entry.length) {
                clearInterval(timerRef.current)
            }
        }, 28) // ~28 ms per character ≈ comfortable reading speed

        return () => clearInterval(timerRef.current)
    }, [dex_entry])

    const typeColor = TYPE_COLORS[type] ?? TYPE_COLORS.Unknown
    const confidencePct = Math.round(confidence * 100)

    return (
        <motion.div
            className="dex-entry"
            initial={{ x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: '100%', opacity: 0 }}
            transition={{ type: 'spring', stiffness: 200, damping: 25 }}
        >
            {/* ── Header ── */}
            <div className="dex-header">
                <h1 className="dex-species">{display_name}</h1>
                <span className="dex-type-badge" style={{ background: typeColor }}>
                    {type}
                </span>
            </div>

            {/* ── Confidence bar ── */}
            <div className="dex-confidence-wrap">
                <span className="dex-confidence-label">Match</span>
                <div className="dex-confidence-track">
                    <motion.div
                        className="dex-confidence-fill"
                        style={{ background: typeColor }}
                        initial={{ width: 0 }}
                        animate={{ width: `${confidencePct}%` }}
                        transition={{ duration: 1, ease: 'easeOut', delay: 0.3 }}
                    />
                </div>
                <span className="dex-confidence-pct">{confidencePct}%</span>
            </div>

            {/* ── Divider ── */}
            <div className="dex-divider" />

            {/* ── Lore text (typewriter) ── */}
            <div className="dex-lore-wrap">
                <span className="dex-lore-title">DEX ENTRY</span>
                <p className="dex-lore-text">
                    {displayedText}
                    {displayedText.length < dex_entry.length && (
                        <span className="dex-cursor">█</span>
                    )}
                </p>
            </div>

            {/* ── Scan again ── */}
            <button className="btn-scan-again" onClick={onScanAgain}>
                ↩ Scan Again
            </button>
        </motion.div>
    )
}
