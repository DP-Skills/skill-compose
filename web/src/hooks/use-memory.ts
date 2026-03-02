'use client';

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { memoryApi } from '@/lib/api';

// Query keys
export const memoryKeys = {
  all: ['memory'] as const,
  files: () => [...memoryKeys.all, 'files'] as const,
  fileList: (agentId?: string) => [...memoryKeys.files(), agentId] as const,
  file: (scope: string, filename: string) => [...memoryKeys.files(), scope, filename] as const,
  entries: () => [...memoryKeys.all, 'entries'] as const,
  entryList: (params: Record<string, unknown>) => [...memoryKeys.entries(), params] as const,
  search: () => [...memoryKeys.all, 'search'] as const,
};

// Bootstrap files
export function useBootstrapFiles(agentId?: string) {
  return useQuery({
    queryKey: memoryKeys.fileList(agentId),
    queryFn: () => memoryApi.listFiles(agentId),
  });
}

export function useBootstrapFile(scope: string, filename: string) {
  return useQuery({
    queryKey: memoryKeys.file(scope, filename),
    queryFn: () => memoryApi.readFile(scope, filename),
    enabled: !!scope && !!filename,
    retry: (count, error: any) => error?.status !== 404 && count < 2,
  });
}

export function useUpdateBootstrapFile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { scope: string; filename: string; content: string }) =>
      memoryApi.writeFile(data.scope, data.filename, data.content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoryKeys.files() });
    },
  });
}

export function useDeleteBootstrapFile() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { scope: string; filename: string }) =>
      memoryApi.deleteFile(data.scope, data.filename),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoryKeys.files() });
    },
  });
}

// Memory entries
export function useMemoryEntries(params?: {
  agent_id?: string;
  category?: string;
  limit?: number;
  offset?: number;
}) {
  return useQuery({
    queryKey: memoryKeys.entryList(params || {}),
    queryFn: () => memoryApi.listEntries(params),
  });
}

export function useCreateMemoryEntry() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (data: { content: string; agent_id?: string; category?: string; source?: string }) =>
      memoryApi.createEntry(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoryKeys.entries() });
    },
  });
}

export function useDeleteMemoryEntry() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => memoryApi.deleteEntry(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: memoryKeys.entries() });
    },
  });
}

export function useMemorySearch() {
  return useMutation({
    mutationFn: (data: { query: string; agent_id?: string; top_k?: number }) =>
      memoryApi.search(data),
  });
}
