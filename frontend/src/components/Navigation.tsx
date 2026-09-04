export type TabType = 'stream' | 'evaluation' | 'audit'

interface NavigationProps {
  activeTab: TabType
  onSelectTab: (tab: TabType) => void
}

export function Navigation({ activeTab, onSelectTab }: NavigationProps) {
  const tabs: { id: TabType; label: string; icon: string }[] = [
    { id: 'stream', label: 'Stream Monitor', icon: '📊' },
    { id: 'evaluation', label: 'Evaluation Benchmarks', icon: '📈' },
    { id: 'audit', label: 'Audit Trail', icon: '📜' },
  ]

  return (
    <nav className="bg-bg-secondary/60 border-b border-border px-6 pt-2">
      <div className="flex space-x-1">
        {tabs.map((tab) => {
          const isActive = activeTab === tab.id
          return (
            <button
              key={tab.id}
              onClick={() => onSelectTab(tab.id)}
              className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-all ${
                isActive
                  ? 'border-accent text-accent bg-bg-card/40'
                  : 'border-transparent text-text-secondary hover:text-text-primary hover:bg-bg-card/20'
              }`}
            >
              <span>{tab.icon}</span>
              <span>{tab.label}</span>
            </button>
          )
        })}
      </div>
    </nav>
  )
}
