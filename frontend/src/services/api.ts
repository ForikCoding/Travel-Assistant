import axios from 'axios'
import type { TripFormData, TripPlanResponse } from '@/types'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 300000,  // 5分钟超时，行程规划需要较长时间
  headers: {
    'Content-Type': 'application/json',
  },
})

// 请求拦截器
apiClient.interceptors.request.use(
  (config) => {
    console.log('发送请求:', config.method?.toUpperCase(), config.url)
    return config
  },
  (error) => {
    console.error('请求错误:', error)
    return Promise.reject(error)
  }
)

// 响应拦截器
apiClient.interceptors.response.use(
  (response) => {
    console.log('收到响应:', response.status, response.data)
    return response
  },
  (error) => {
    console.error('响应错误:', error)
    return Promise.reject(error)
  }
)

// 生成旅行计划
export async function generateTripPlan(formData: TripFormData): Promise<TripPlanResponse> {
  try {
    const response = await apiClient.post<TripPlanResponse>('/api/trip/plan', formData)
    return response.data
  } catch (error) {
    console.error('生成旅行计划失败:', error)
    throw error
  }
}

// 健康检查
export async function healthCheck(): Promise<any> {
    try {
        const response = await apiClient.get('/api/health')
        return response.data
    } catch (error: any) {
        console.error('健康检查失败:', error)
        throw new Error(error.message || '健康检查失败')
    }
}

export default apiClient
