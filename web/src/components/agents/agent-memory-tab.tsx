'use client';

import { useState, useCallback, useEffect, useRef } from 'react';
import { Plus, Search, Trash2, Save, Brain } from 'lucide-react';
import { toast } from 'sonner';
import type { MemoryEntry, MemorySearchResult } from '@/lib/api';
import { useTranslation } from '@/i18n/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge, type BadgeProps } from '@/components/ui/badge';
import { Spinner } from '@/components/ui/spinner';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from '@/components/ui/alert-dialog';
import {
  useBootstrapFiles,
  useBootstrapFile,
  useUpdateBootstrapFile,
  useDeleteBootstrapFile,
  useMemoryEntries,
  useCreateMemoryEntry,
  useDeleteMemoryEntry,
  useMemorySearch,
} from '@/hooks/use-memory';

const BOOTSTRAP_FILES = ['SOUL.md', 'USER.md', 'MEMORY.md'];

const CATEGORY_COLORS: Record<string, BadgeProps['variant']> = {
  fact: 'info',
  preference: 'purple',
  procedure: 'success',
  context: 'secondary',
  session_summary: 'warning',
};

interface AgentMemoryTabProps {
  agentId: string;
}

export function AgentMemoryTab({ agentId }: AgentMemoryTabProps) {
  const { t } = useTranslation('agents');
  const { t: tc } = useTranslation('common');

  return (
    <div className="space-y-6">
      <BootstrapFileEditor agentId={agentId} />
      <MemoryEntriesList agentId={agentId} />
    </div>
  );
}

// ─── Bootstrap File Editor ──────────────────────────────────────

function BootstrapFileEditor({ agentId }: { agentId: string }) {
  const { t } = useTranslation('agents');
  const { t: tc } = useTranslation('common');
  const [selectedFile, setSelectedFile] = useState(BOOTSTRAP_FILES[0]);
  const [scope, setScope] = useState<'global' | 'agent'>('global');
  const [editorContent, setEditorContent] = useState('');
  const [isDirty, setIsDirty] = useState(false);

  const effectiveScope = scope === 'agent' ? agentId : 'global';
  const { data: filesData } = useBootstrapFiles(agentId);
  const { data: fileData, isLoading: isLoadingFile, isFetching, isError } = useBootstrapFile(effectiveScope, selectedFile);
  const updateFile = useUpdateBootstrapFile();
  const deleteFile = useDeleteBootstrapFile();

  // Clear editor immediately when scope or file changes to avoid showing stale data
  useEffect(() => {
    setEditorContent('');
    setIsDirty(false);
  }, [selectedFile, scope]);

  // Sync editor content when file data arrives (clear on 404/error)
  useEffect(() => {
    if (!isFetching) {
      setEditorContent(isError ? '' : (fileData?.content ?? ''));
      setIsDirty(false);
    }
  }, [fileData?.content, isFetching, isError]);

  const handleSave = useCallback(async () => {
    try {
      await updateFile.mutateAsync({
        scope: effectiveScope,
        filename: selectedFile,
        content: editorContent,
      });
      setIsDirty(false);
      toast.success(t('memory.fileSaved'));
    } catch {
      toast.error(t('memory.fileSaveError'));
    }
  }, [updateFile, effectiveScope, selectedFile, editorContent, t]);

  const handleDelete = useCallback(async () => {
    try {
      await deleteFile.mutateAsync({
        scope: effectiveScope,
        filename: selectedFile,
      });
      setEditorContent('');
      setIsDirty(false);
      toast.success(t('memory.fileDeleted'));
    } catch {
      toast.error(t('memory.fileDeleteError'));
    }
  }, [deleteFile, effectiveScope, selectedFile, t]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Brain className="h-4 w-4" />
          {t('memory.bootstrapFiles')}
        </CardTitle>
        <p className="text-sm text-muted-foreground">{t('memory.bootstrapDescription')}</p>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* File tabs */}
        <div className="flex items-center gap-2">
          {BOOTSTRAP_FILES.map((file) => {
            const info = filesData?.files?.find(f => f.filename === file);
            const hasContent = scope === 'agent'
              ? info?.agent_exists
              : info?.global_exists;
            return (
              <button
                key={file}
                onClick={() => { setSelectedFile(file); setIsDirty(false); }}
                className={`px-3 py-1.5 text-sm rounded-md transition-colors ${
                  selectedFile === file
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted hover:bg-muted/80 text-muted-foreground'
                }`}
              >
                {file}
                {hasContent && <span className="ml-1 text-xs opacity-70">*</span>}
              </button>
            );
          })}
        </div>

        {/* Scope toggle */}
        <div className="flex items-center gap-2 text-sm">
          <span className="text-muted-foreground">{t('memory.scope')}:</span>
          <button
            onClick={() => setScope('global')}
            className={`px-2 py-1 rounded text-xs ${
              scope === 'global'
                ? 'bg-primary/10 text-primary font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {t('memory.scopeGlobal')}
          </button>
          <button
            onClick={() => setScope('agent')}
            className={`px-2 py-1 rounded text-xs ${
              scope === 'agent'
                ? 'bg-primary/10 text-primary font-medium'
                : 'text-muted-foreground hover:text-foreground'
            }`}
          >
            {t('memory.scopeAgent')}
          </button>
        </div>

        {/* Editor */}
        {isLoadingFile ? (
          <div className="flex items-center justify-center py-8">
            <Spinner size="sm" />
          </div>
        ) : (
          <textarea
            value={editorContent}
            onChange={(e) => { setEditorContent(e.target.value); setIsDirty(true); }}
            placeholder={t('memory.editorPlaceholder', { filename: selectedFile })}
            className="w-full h-48 p-3 text-sm font-mono border rounded-md bg-background resize-y focus:outline-none focus:ring-2 focus:ring-ring"
          />
        )}

        {/* Actions */}
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            onClick={handleSave}
            disabled={!isDirty || updateFile.isPending}
          >
            {updateFile.isPending ? <Spinner size="sm" className="mr-2" /> : <Save className="h-3.5 w-3.5 mr-1.5" />}
            {tc('actions.save')}
          </Button>
          {fileData?.content && (
            <Button
              size="sm"
              variant="outline"
              onClick={handleDelete}
              disabled={deleteFile.isPending}
              className="text-destructive hover:text-destructive"
            >
              <Trash2 className="h-3.5 w-3.5 mr-1.5" />
              {tc('actions.delete')}
            </Button>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

// ─── Memory Entries List ────────────────────────────────────────

function MemoryEntriesList({ agentId }: { agentId: string }) {
  const { t } = useTranslation('agents');
  const { t: tc } = useTranslation('common');
  const [searchQuery, setSearchQuery] = useState('');
  const [isSearchMode, setIsSearchMode] = useState(false);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState<string | null>(null);
  const [newContent, setNewContent] = useState('');
  const [newCategory, setNewCategory] = useState('fact');

  const { data: entriesData, isLoading } = useMemoryEntries({ agent_id: agentId });
  const createEntry = useCreateMemoryEntry();
  const deleteEntry = useDeleteMemoryEntry();
  const searchMemory = useMemorySearch();
  const searchMemoryRef = useRef(searchMemory);
  searchMemoryRef.current = searchMemory;

  const entries = isSearchMode && searchMemory.data
    ? searchMemory.data.results
    : entriesData?.entries || [];

  const handleSearch = useCallback(() => {
    if (!searchQuery.trim()) {
      setIsSearchMode(false);
      return;
    }
    setIsSearchMode(true);
    searchMemoryRef.current.mutate({ query: searchQuery, agent_id: agentId, top_k: 20 });
  }, [searchQuery, agentId]);

  const handleClearSearch = useCallback(() => {
    setSearchQuery('');
    setIsSearchMode(false);
  }, []);

  const handleAdd = useCallback(async () => {
    if (!newContent.trim()) return;
    try {
      await createEntry.mutateAsync({
        content: newContent,
        agent_id: agentId,
        category: newCategory,
        source: 'manual',
      });
      setShowAddDialog(false);
      setNewContent('');
      setNewCategory('fact');
      toast.success(t('memory.entrySaved'));
    } catch {
      toast.error(t('memory.entrySaveError'));
    }
  }, [createEntry, newContent, agentId, newCategory, t]);

  const handleDelete = useCallback(async (id: string) => {
    try {
      await deleteEntry.mutateAsync(id);
      setShowDeleteDialog(null);
      toast.success(t('memory.entryDeleted'));
    } catch {
      toast.error(t('memory.entryDeleteError'));
    }
  }, [deleteEntry, t]);

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-base">{t('memory.entries')}</CardTitle>
          <Button size="sm" variant="outline" onClick={() => setShowAddDialog(true)}>
            <Plus className="h-3.5 w-3.5 mr-1.5" />
            {t('memory.addEntry')}
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {/* Search bar */}
        <div className="flex gap-2">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              placeholder={t('memory.searchPlaceholder')}
              className="w-full pl-9 pr-3 py-2 text-sm border rounded-md bg-background focus:outline-none focus:ring-2 focus:ring-ring"
            />
          </div>
          <Button size="sm" onClick={handleSearch} disabled={searchMemory.isPending}>
            {searchMemory.isPending ? <Spinner size="sm" /> : <Search className="h-4 w-4" />}
          </Button>
          {isSearchMode && (
            <Button size="sm" variant="ghost" onClick={handleClearSearch}>
              {tc('actions.clear')}
            </Button>
          )}
        </div>

        {/* Entries list */}
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Spinner size="sm" />
          </div>
        ) : entries.length === 0 ? (
          <div className="text-center py-8 text-sm text-muted-foreground">
            {isSearchMode ? t('memory.noSearchResults') : t('memory.noEntries')}
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {entries.map((entry: MemoryEntry | MemorySearchResult) => (
              <div
                key={entry.id}
                className="flex items-start gap-2 p-3 rounded-md border bg-muted/30 hover:bg-muted/50 transition-colors"
              >
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {entry.category && (
                      <Badge variant={CATEGORY_COLORS[entry.category] || 'secondary'} className="text-xs">
                        {entry.category}
                      </Badge>
                    )}
                    {entry.source && (
                      <span className="text-xs text-muted-foreground">{entry.source}</span>
                    )}
                    {entry.similarity != null && (
                      <span className="text-xs text-muted-foreground ml-auto">
                        {(entry.similarity * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <p className="text-sm break-words">{entry.content}</p>
                </div>
                <button
                  onClick={() => setShowDeleteDialog(entry.id)}
                  className="shrink-0 p-1 text-muted-foreground hover:text-destructive transition-colors"
                  title={tc('actions.delete')}
                >
                  <Trash2 className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
          </div>
        )}
      </CardContent>

      {/* Add Entry Dialog */}
      <AlertDialog open={showAddDialog} onOpenChange={setShowAddDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('memory.addEntry')}</AlertDialogTitle>
            <AlertDialogDescription>{t('memory.addEntryDescription')}</AlertDialogDescription>
          </AlertDialogHeader>
          <div className="space-y-3 py-2">
            <textarea
              value={newContent}
              onChange={(e) => setNewContent(e.target.value)}
              placeholder={t('memory.entryContentPlaceholder')}
              className="w-full h-24 p-3 text-sm border rounded-md bg-background resize-y focus:outline-none focus:ring-2 focus:ring-ring"
            />
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">{t('memory.category')}:</span>
              {['fact', 'preference', 'procedure', 'context'].map((cat) => (
                <button
                  key={cat}
                  onClick={() => setNewCategory(cat)}
                  className={`px-2 py-1 rounded text-xs transition-colors ${
                    newCategory === cat
                      ? 'bg-primary/10 text-primary font-medium'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {cat}
                </button>
              ))}
            </div>
          </div>
          <AlertDialogFooter>
            <AlertDialogCancel>{tc('actions.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleAdd}
              disabled={!newContent.trim() || createEntry.isPending}
            >
              {createEntry.isPending ? <Spinner size="sm" className="mr-2" /> : null}
              {tc('actions.save')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Delete Confirmation Dialog */}
      <AlertDialog open={!!showDeleteDialog} onOpenChange={() => setShowDeleteDialog(null)}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('memory.deleteEntry')}</AlertDialogTitle>
            <AlertDialogDescription>{t('memory.deleteEntryConfirm')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{tc('actions.cancel')}</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => showDeleteDialog && handleDelete(showDeleteDialog)}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              {tc('actions.delete')}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </Card>
  );
}
