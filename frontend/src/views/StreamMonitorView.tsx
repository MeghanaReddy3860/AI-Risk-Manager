import { useEffect, useState } from 'react'
import { getMerchants, getMerchantTimeseries, getWindows } from '../api'
import { DetectionWindowsTable } from '../components/DetectionWindowsTable'
import { MerchantTimeseriesChart } from '../components/MerchantTimeseriesChart'
import { OverviewCards } from '../components/OverviewCards'
import { WindowDetailModal } from '../components/WindowDetailModal'
import type { DetectionWindow, MerchantSummary, TimeseriesPoint } from '../types'

interface StreamMonitorViewProps {
  refreshTrigger?: number
  onInspectWindow?: (windowId: number) => void
}

export function StreamMonitorView({ refreshTrigger, onInspectWindow }: StreamMonitorViewProps) {
  // Modal state for Window Deep Dive
  const [inspectWindowId, setInspectWindowId] = useState<number | null>(null)
  const [isModalOpen, setIsModalOpen] = useState<boolean>(false)
  // Merchant summaries & KPI state
  const [merchants, setMerchants] = useState<MerchantSummary[]>([])
  const [merchantsLoading, setMerchantsLoading] = useState(true)
  const [merchantsError, setMerchantsError] = useState<string | null>(null)

  // Selected merchant & Timeseries state
  const [selectedMerchant, setSelectedMerchant] = useState<string>('')
  const [timeseries, setTimeseries] = useState<TimeseriesPoint[]>([])
  const [timeseriesLoading, setTimeseriesLoading] = useState(true)
  const [timeseriesError, setTimeseriesError] = useState<string | null>(null)

  // Detection windows table state & filters
  const [windows, setWindows] = useState<DetectionWindow[]>([])
  const [windowsLoading, setWindowsLoading] = useState(true)
  const [windowsError, setWindowsError] = useState<string | null>(null)
  const [filterMerchant, setFilterMerchant] = useState<string>('')
  const [filterSplit, setFilterSplit] = useState<string>('')
  const [filterFlagged, setFilterFlagged] = useState<boolean>(false)
  const [page, setPage] = useState<number>(1)
  const limit = 50

  // 1. Initial Load of Merchant Summaries
  useEffect(() => {
    let isMounted = true

    getMerchants()
      .then((data) => {
        if (isMounted) {
          setMerchants(data)
          setMerchantsError(null)
          if (data.length > 0) {
            setSelectedMerchant((prev) => prev || data[0].merchant_id)
          }
          setMerchantsLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to load merchants'
          setMerchantsError(msg)
          setMerchantsLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [refreshTrigger])

  // 2. Fetch Timeseries whenever selectedMerchant changes or refreshTrigger fires
  useEffect(() => {
    if (!selectedMerchant) return
    let isMounted = true

    getMerchantTimeseries(selectedMerchant)
      .then((data) => {
        if (isMounted) {
          setTimeseries(data)
          setTimeseriesError(null)
          setTimeseriesLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to load timeseries'
          setTimeseriesError(msg)
          setTimeseries([])
          setTimeseriesLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [selectedMerchant, refreshTrigger])

  // 3. Fetch Detection Windows whenever filters, page, or refreshTrigger changes
  useEffect(() => {
    let isMounted = true

    const offset = (page - 1) * limit
    getWindows({
      merchant_id: filterMerchant || undefined,
      split: filterSplit || undefined,
      is_flagged: filterFlagged ? true : undefined,
      limit,
      offset,
    })
      .then((data) => {
        if (isMounted) {
          setWindows(data)
          setWindowsError(null)
          setWindowsLoading(false)
        }
      })
      .catch((err: unknown) => {
        if (isMounted) {
          const msg = err instanceof Error ? err.message : 'Failed to load detection windows'
          setWindowsError(msg)
          setWindows([])
          setWindowsLoading(false)
        }
      })

    return () => {
      isMounted = false
    }
  }, [filterMerchant, filterSplit, filterFlagged, page, limit, refreshTrigger])

  const handleMerchantSelect = (id: string) => {
    setSelectedMerchant(id)
    setTimeseriesLoading(true)
  }

  const handleFilterMerchantChange = (id: string) => {
    setFilterMerchant(id)
    setPage(1)
    setWindowsLoading(true)
  }

  const handleFilterSplitChange = (split: string) => {
    setFilterSplit(split)
    setPage(1)
    setWindowsLoading(true)
  }

  const handleFilterFlaggedToggle = (flagged: boolean) => {
    setFilterFlagged(flagged)
    setPage(1)
    setWindowsLoading(true)
  }

  const handlePageChange = (newPage: number) => {
    setPage(newPage)
    setWindowsLoading(true)
  }

  const handleInspectWindow = (windowId: number) => {
    setInspectWindowId(windowId)
    setIsModalOpen(true)
    if (onInspectWindow) {
      onInspectWindow(windowId)
    }
  }

  return (
    <div className="space-y-6">
      {/* 1. Overview KPI Cards */}
      <OverviewCards merchants={merchants} loading={merchantsLoading} error={merchantsError} />

      {/* 2. Merchant Activity Stream Timeseries Chart */}
      <MerchantTimeseriesChart
        merchantId={selectedMerchant}
        merchants={merchants}
        onSelectMerchant={handleMerchantSelect}
        data={timeseries}
        loading={timeseriesLoading}
        error={timeseriesError}
      />

      {/* 3. Detection Windows Stream Table */}
      <DetectionWindowsTable
        windows={windows}
        loading={windowsLoading}
        error={windowsError}
        merchants={merchants}
        selectedMerchant={filterMerchant}
        selectedSplit={filterSplit}
        flaggedOnly={filterFlagged}
        page={page}
        limit={limit}
        onMerchantChange={handleFilterMerchantChange}
        onSplitChange={handleFilterSplitChange}
        onFlaggedToggle={handleFilterFlaggedToggle}
        onPageChange={handlePageChange}
        onInspectWindow={handleInspectWindow}
      />

      {/* 4. Window Deep Dive Investigation Drawer Modal */}
      <WindowDetailModal
        windowId={inspectWindowId}
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
      />
    </div>
  )
}
