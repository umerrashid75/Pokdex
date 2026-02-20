/**
 * PokeBallIntro.jsx
 * Full-screen Pokéball opening animation using Framer Motion.
 *
 * Phases:
 *   waiting → (user taps) → idle → spin (+audio) → split → fade → onComplete
 *
 * Audio: place your clip at  frontend/public/whos-that-pokemon.mp3
 */

import { motion, AnimatePresence } from 'framer-motion'
import { useState, useEffect, useRef, useCallback } from 'react'

export default function PokeBallIntro({ onComplete }) {
    // 'waiting' = "Tap to Start" splash
    // 'idle'    = ball appears, tagline fades in
    // 'spin'    = 720° rotation + audio
    // 'split'   = halves fly apart + flash
    // 'fade'    = full screen fades to black
    const [phase, setPhase] = useState('waiting')
    const audioRef = useRef(null)
    const timersRef = useRef([])

    // Pre-load audio on mount
    useEffect(() => {
        const audio = new Audio('/whos-that-pokemon.mp3')
        audio.preload = 'auto'
        audio.volume = 0.85
        audioRef.current = audio
        return () => { audio.pause(); audio.src = '' }
    }, [])

    // Clear any running timers on unmount
    useEffect(() => () => timersRef.current.forEach(clearTimeout), [])

    const startSequence = useCallback(() => {
        if (phase !== 'waiting') return   // guard against double-tap

        setPhase('idle')

        const after = (ms, fn) => {
            const id = setTimeout(fn, ms)
            timersRef.current.push(id)
        }

        after(400, () => {
            setPhase('spin')
            // Play audio – unlocked because this runs inside a user-gesture callback chain
            if (audioRef.current) {
                audioRef.current.currentTime = 0
                audioRef.current.play().catch(() => { })
            }
        })
        after(1600, () => setPhase('split'))
        after(2800, () => setPhase('fade'))
        after(3600, () => onComplete())
    }, [phase, onComplete])

    const isSplitOrFade = phase === 'split' || phase === 'fade'

    return (
        <AnimatePresence>
            {phase !== 'done' && (
                <motion.div
                    className="pokeball-intro-backdrop"
                    initial={{ opacity: 1 }}
                    animate={{ opacity: phase === 'fade' ? 0 : 1 }}
                    transition={{ duration: 0.8, ease: 'easeInOut' }}
                    onClick={startSequence}
                    style={{ cursor: phase === 'waiting' ? 'pointer' : 'default' }}
                >

                    {/* ── "Tap to Start" overlay ── */}
                    <AnimatePresence>
                        {phase === 'waiting' && (
                            <motion.div
                                key="tap-overlay"
                                className="tap-to-start"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                exit={{ opacity: 0 }}
                                transition={{ duration: 0.5 }}
                            >
                                {/* Idle pokéball silhouette behind the text */}
                                <div className="tap-pokeball-bg" />
                                <motion.span
                                    className="tap-label"
                                    animate={{ opacity: [1, 0.35, 1] }}
                                    transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
                                >
                                    TAP TO START
                                </motion.span>
                            </motion.div>
                        )}
                    </AnimatePresence>

                    {/* ── Spinning Pokéball ── */}
                    {phase !== 'waiting' && (
                        <motion.div
                            className="pokeball-wrap"
                            initial={{ opacity: 0, scale: 0.7 }}
                            animate={
                                phase === 'idle'
                                    ? { opacity: 1, scale: 1, rotate: 0 }
                                    : phase === 'spin'
                                        ? { opacity: 1, rotate: 720, scale: 1.06 }
                                        : isSplitOrFade
                                            ? { opacity: 1, rotate: 720, scale: 1 }
                                            : {}
                            }
                            transition={{
                                opacity: { duration: 0.4 },
                                rotate: { duration: 1.1, ease: [0.4, 0, 0.2, 1] },
                                scale: { duration: 0.4, ease: 'easeOut' },
                            }}
                        >
                            {/* Top half */}
                            <motion.div
                                className="pokeball-half pokeball-top"
                                animate={isSplitOrFade ? { y: -160, rotate: -15 } : { y: 0, rotate: 0 }}
                                transition={{ type: 'spring', stiffness: 120, damping: 14, delay: 0.05 }}
                            >
                                <div className="pokeball-top-inner" />
                            </motion.div>

                            {/* Centre band + button */}
                            <motion.div
                                className="pokeball-band"
                                animate={isSplitOrFade ? { scaleX: 1.08 } : { scaleX: 1 }}
                                transition={{ duration: 0.3 }}
                            >
                                <motion.div
                                    className="pokeball-button"
                                    animate={
                                        phase === 'spin'
                                            ? { boxShadow: ['0 0 0px #fff', '0 0 22px #fff', '0 0 0px #fff'] }
                                            : phase === 'split'
                                                ? { scale: [1, 1.6, 1], boxShadow: ['0 0 0px #fff', '0 0 34px #fff', '0 0 0px #fff'] }
                                                : {}
                                    }
                                    transition={{ duration: phase === 'spin' ? 1.1 : 0.6, ease: 'easeInOut' }}
                                />
                            </motion.div>

                            {/* Bottom half */}
                            <motion.div
                                className="pokeball-half pokeball-bottom"
                                animate={isSplitOrFade ? { y: 160, rotate: 15 } : { y: 0, rotate: 0 }}
                                transition={{ type: 'spring', stiffness: 120, damping: 14, delay: 0.05 }}
                            >
                                <div className="pokeball-bottom-inner" />
                            </motion.div>

                            {/* Flash burst on split */}
                            {phase === 'split' && (
                                <motion.div
                                    className="pokeball-flash"
                                    initial={{ opacity: 0, scale: 0.5 }}
                                    animate={{ opacity: [0, 1, 0], scale: [0.5, 2.5, 3] }}
                                    transition={{ duration: 0.9, ease: 'easeOut' }}
                                />
                            )}
                        </motion.div>
                    )}

                    {/* Tagline – visible during idle + spin */}
                    {phase !== 'waiting' && (
                        <motion.p
                            className="pokeball-tagline"
                            initial={{ opacity: 0, y: 20 }}
                            animate={{
                                opacity: phase === 'idle' || phase === 'spin' ? 1 : 0,
                                y: phase === 'idle' || phase === 'spin' ? 0 : -10,
                            }}
                            transition={{ delay: 0.2, duration: 0.5 }}
                        >
                            Smart Pokédex
                        </motion.p>
                    )}

                </motion.div>
            )}
        </AnimatePresence>
    )
}
