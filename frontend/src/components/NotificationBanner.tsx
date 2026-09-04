interface NotificationBannerProps {
  type: 'success' | 'error' | 'info'
  message: string
  onClose?: () => void
}

export function NotificationBanner({ type, message, onClose }: NotificationBannerProps) {
  const bgColor =
    type === 'success'
      ? 'bg-status-low/10 border-status-low/30 text-status-low'
      : type === 'error'
      ? 'bg-status-critical/10 border-status-critical/30 text-status-critical'
      : 'bg-accent/10 border-accent/30 text-accent'

  const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'

  return (
    <div className={`border rounded-lg p-4 flex items-center justify-between shadow-sm transition-all ${bgColor}`}>
      <div className="flex items-center gap-3">
        <span className="text-lg">{icon}</span>
        <span className="text-sm font-medium">{message}</span>
      </div>
      {onClose && (
        <button
          onClick={onClose}
          className="text-text-muted hover:text-text-primary text-sm font-semibold px-2 py-1 rounded transition-colors"
          aria-label="Close notification"
        >
          ✕
        </button>
      )}
    </div>
  )
}
