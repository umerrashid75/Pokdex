/**
 * StatusLED.jsx
 * A small blinking LED indicator that pulses while scanning is in progress.
 */

import { motion, AnimatePresence } from 'framer-motion'

export default function StatusLED({ isScanning }) {
    return (
        <div className="status-led-wrap">
            <AnimatePresence>
                {isScanning ? (
                    <motion.div
                        key="scanning"
                        className="status-led scanning"
                        animate={{ opacity: [1, 0.2, 1], scale: [1, 0.85, 1] }}
                        transition={{ duration: 0.7, repeat: Infinity, ease: 'easeInOut' }}
                    />
                ) : (
                    <motion.div
                        key="idle"
                        className="status-led idle"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ duration: 0.3 }}
                    />
                )}
            </AnimatePresence>
            <span className="status-led-label">
                {isScanning ? 'PROCESSING' : 'STANDBY'}
            </span>
        </div>
    )
}
