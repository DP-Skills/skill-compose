'use client';

import { useState, useMemo } from 'react';
import { useRouter, useParams, useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { useQuery } from '@tanstack/react-query';
import { ArrowLeft, MessageSquare, Bot, Server } from 'lucide-react';
import { Spinner } from '@/components/ui/spinner';
import { useTranslation } from '@/i18n/client';
import { getAgentDescription, getAgentDisplayName } from '@/lib/seed-descriptions';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
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
import { useAgentPreset, useUpdateAgentPreset, useDeleteAgentPreset, usePublishAgent, useUnpublishAgent } from '@/hooks/use-agents';
import { useExecutors } from '@/hooks/use-executors';
import { modelsApi, toolsApi } from '@/lib/api';
import { useChatStore } from '@/stores/chat-store';
import { useChatPanel } from '@/components/chat/chat-provider';
import { AgentConfigForm, type AgentFormValues } from '@/components/agents/agent-config-form';
import { PublishCard } from '@/components/agents/publish-card';
import { PublishDialog } from '@/components/agents/publish-dialog';
import { AgentMemoryTab } from '@/components/agents/agent-memory-tab';

export default function AgentDetailPage() {
  const router = useRouter();
  const params = useParams();
  const searchParams = useSearchParams();
  const presetId = params.id as string;
  const initialTab = searchParams.get('tab') || 'overview';

  const { data: preset, isLoading, error } = useAgentPreset(presetId);
  const updatePreset = useUpdateAgentPreset(presetId);
  const deletePreset = useDeleteAgentPreset();
  const publishAgent = usePublishAgent(presetId);
  const unpublishAgent = useUnpublishAgent(presetId);

  const { t } = useTranslation('agents');
  const { t: tc } = useTranslation('common');

  const [showPublishDialog, setShowPublishDialog] = useState(false);
  const [isPublishing, setIsPublishing] = useState(false);
  const [showClearConversationDialog, setShowClearConversationDialog] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);
  const [formErrors, setFormErrors] = useState<string | null>(null);

  // Fetch models for publish dialog
  const { data: modelsData } = useQuery({
    queryKey: ['models-providers'],
    queryFn: () => modelsApi.listProviders(),
  });
  const modelProviders = modelsData?.providers || [];

  // Fetch tools for apply-to-chat
  const { data: toolsData } = useQuery({
    queryKey: ['tools-list'],
    queryFn: () => toolsApi.list(),
  });
  const tools = toolsData?.tools || [];

  // Fetch executors for status display
  const { data: executorsData } = useExecutors();
  const executorInfo = useMemo(() => {
    if (!preset?.executor_name || !executorsData?.executors) return null;
    return executorsData.executors.find(e => e.name === preset.executor_name) || null;
  }, [preset?.executor_name, executorsData?.executors]);
  const isExecutorOffline = executorInfo ? executorInfo.status !== 'online' : false;

  // Chat store + panel
  const {
    messages,
    setSelectedSkills,
    setSelectedTools,
    setSelectedMcpServers,
    setMaxTurns,
    setSelectedAgentPreset,
    setSelectedModel,
    setSelectedExecutorName: setChatExecutorId,
    clearMessages,
    clearUploadedFiles,
    setSessionId,
  } = useChatStore();
  const chatPanel = useChatPanel();

  // Memoize initialValues so the form's sync-useEffect doesn't reset user edits
  // on every parent re-render. Only recompute when the preset actually changes.
  const formInitialValues = useMemo(() => {
    if (!preset) return null;
    return {
      name: preset.name,
      description: preset.description || '',
      system_prompt: preset.system_prompt || '',
      skill_ids: preset.skill_ids || [],
      builtin_tools: preset.builtin_tools ?? undefined,
      mcp_servers: preset.mcp_servers || [],
      max_turns: preset.max_turns,
      model_provider: preset.model_provider || null,
      model_name: preset.model_name || null,
      executor_name: preset.executor_name || null,
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [preset?.id, preset?.updated_at]);

  // ─── Handlers ──────────────────────────────────────────

  const handleTabChange = (value: string) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set('tab', value);
    router.replace(`?${params.toString()}`, { scroll: false });
  };

  const applyPresetToChat = () => {
    if (!preset) return;
    // Reset session so the new agent gets a fresh server-side session
    setSessionId(null);
    setSelectedSkills(preset.skill_ids || []);
    setMaxTurns(preset.max_turns);
    if (preset.builtin_tools === null && tools.length > 0) {
      setSelectedTools(tools.map(t => t.name));
    } else if (preset.builtin_tools && preset.builtin_tools.length > 0) {
      setSelectedTools(preset.builtin_tools);
    } else {
      setSelectedTools([]);
    }
    setSelectedMcpServers(preset.mcp_servers || []);
    setSelectedModel(preset.model_provider || null, preset.model_name || null);
    setChatExecutorId(preset.executor_name || null);
    setSelectedAgentPreset(preset.id);
    chatPanel.open(preset.skill_ids || []);
  };

  const handleApplyToChat = () => {
    if (!preset) return;
    if (messages.length > 0) {
      setShowClearConversationDialog(true);
    } else {
      applyPresetToChat();
    }
  };

  const handleConfirmClearAndApply = () => {
    clearMessages();
    clearUploadedFiles();
    applyPresetToChat();
    setShowClearConversationDialog(false);
  };

  const handleSave = async (values: AgentFormValues) => {
    if (!preset) return;
    setFormErrors(null);

    const allToolsSelected = tools.length > 0 && values.builtin_tools.length === tools.length;
    const builtinToolsToSave = allToolsSelected ? null : values.builtin_tools;

    try {
      await updatePreset.mutateAsync({
        name: values.name,
        description: values.description || undefined,
        system_prompt: values.system_prompt || undefined,
        skill_ids: values.skill_ids.length > 0 ? values.skill_ids : [],
        builtin_tools: builtinToolsToSave,
        mcp_servers: values.mcp_servers.length > 0 ? values.mcp_servers : [],
        max_turns: values.max_turns,
        model_provider: values.model_provider ?? null,
        model_name: values.model_name ?? null,
        executor_name: values.executor_name ?? null,
      });
      setHasChanges(false);
    } catch (error) {
      setFormErrors(error instanceof Error ? error.message : t('edit.error'));
    }
  };

  const handleDelete = async () => {
    if (!preset || preset.is_system) return;
    try {
      await deletePreset.mutateAsync(presetId);
      router.push('/agents');
    } catch (error) {
      setFormErrors(error instanceof Error ? error.message : t('delete.error'));
    }
  };

  const handlePublish = async (
    modelProvider: string | null,
    modelName: string | null,
    responseMode: 'streaming' | 'non_streaming'
  ) => {
    if (!preset) return;
    setIsPublishing(true);
    try {
      const modelChanged = modelProvider !== preset.model_provider || modelName !== preset.model_name;
      if (modelChanged) {
        await updatePreset.mutateAsync({
          model_provider: modelProvider ?? null,
          model_name: modelName ?? null,
        });
      }
      await publishAgent.mutateAsync({ api_response_mode: responseMode });
      setShowPublishDialog(false);
    } catch (error) {
      setFormErrors(error instanceof Error ? error.message : t('publish.error'));
    } finally {
      setIsPublishing(false);
    }
  };

  // ─── Loading / Error states ────────────────────────────

  if (isLoading) {
    return (
      <div className="container mx-auto py-8 max-w-2xl">
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" className="text-muted-foreground" />
        </div>
      </div>
    );
  }

  if (error || !preset) {
    return (
      <div className="container mx-auto py-8 max-w-2xl">
        <div className="mb-6">
          <Link href="/agents" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            {t('detail.backToAgents')}
          </Link>
        </div>
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-destructive">
          <p className="font-medium">{t('detail.loadError')}</p>
          <p className="text-sm mt-1">{(error as Error)?.message || t('detail.notFound')}</p>
        </div>
      </div>
    );
  }

  const isProcessing = updatePreset.isPending || deletePreset.isPending;
  const translatedDescription = getAgentDescription(t, preset.name, preset.description);

  // ─── Render ────────────────────────────────────────────

  return (
    <div className="container mx-auto py-8 max-w-2xl">
      {/* Header */}
      <div className="mb-6">
        <Link href="/agents" className="inline-flex items-center text-sm text-muted-foreground hover:text-foreground mb-4">
          <ArrowLeft className="mr-2 h-4 w-4" />
          {t('detail.backToAgents')}
        </Link>
        <div className="flex items-center gap-3">
          <Bot className="h-8 w-8 text-primary" />
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-3xl font-bold">{getAgentDisplayName(t, preset.name)}</h1>
              {preset.is_system && <Badge variant="secondary">{t('type.system')}</Badge>}
            </div>
            {translatedDescription && (
              <p className="text-muted-foreground mt-1">{translatedDescription}</p>
            )}
          </div>
        </div>
      </div>

      {/* Tabs */}
      <Tabs value={initialTab} onValueChange={handleTabChange}>
        <TabsList className="grid w-full grid-cols-3 mb-6">
          <TabsTrigger value="overview">{t('detail.tabs.overview')}</TabsTrigger>
          <TabsTrigger value="configuration">{t('detail.tabs.configuration')}</TabsTrigger>
          <TabsTrigger value="memory">{t('detail.tabs.memory')}</TabsTrigger>
        </TabsList>

        {/* Overview Tab */}
        <TabsContent value="overview" className="space-y-6">
          {/* Apply to Chat */}
          <Card>
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-semibold">{t('detail.testTitle')}</h3>
                  <p className="text-sm text-muted-foreground">{t('detail.testDescription')}</p>
                </div>
                <Button onClick={handleApplyToChat}>
                  <MessageSquare className="mr-2 h-4 w-4" />
                  {t('applyToChat')}
                </Button>
              </div>
            </CardContent>
          </Card>

          {/* Publish */}
          {!preset.is_system && (
            <PublishCard
              preset={preset}
              isUnpublishing={unpublishAgent.isPending}
              executorOffline={isExecutorOffline}
              executorName={executorInfo?.name}
              onPublish={() => setShowPublishDialog(true)}
              onUnpublish={() => unpublishAgent.mutate()}
            />
          )}

          {/* Metadata */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('detail.metadataTitle')}</CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-2">
              <div className="flex justify-between">
                <span>ID:</span>
                <code className="text-xs bg-muted px-1.5 py-0.5 rounded">{preset.id}</code>
              </div>
              <div className="flex justify-between">
                <span>{t('model.title')}:</span>
                <span>
                  {preset.model_provider && preset.model_name
                    ? `${preset.model_provider}/${preset.model_name}`
                    : t('detail.modelDefault')}
                </span>
              </div>
              {executorInfo && (
                <div className="flex justify-between items-center">
                  <span className="flex items-center gap-1">
                    <Server className="h-3.5 w-3.5" />
                    {t('executor.label')}:
                  </span>
                  <Badge variant={executorInfo.status === 'online' ? 'outline-success' : 'outline-error'} className="text-xs">
                    {executorInfo.name} ({executorInfo.status === 'online' ? t('executor.online') : t('executor.offline')})
                  </Badge>
                </div>
              )}
              <div className="flex justify-between">
                <span>{t('overview.createdAt')}:</span>
                <span>{new Date(preset.created_at).toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span>{t('overview.updatedAt')}:</span>
                <span>{new Date(preset.updated_at).toLocaleString()}</span>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Configuration Tab */}
        <TabsContent value="configuration">
          <Card>
            <CardHeader>
              <CardTitle>{t('detail.configuration')}</CardTitle>
            </CardHeader>
            <CardContent>
              <AgentConfigForm
                mode="edit"
                initialValues={formInitialValues!}
                isSystem={preset.is_system}
                isProcessing={isProcessing}
                isSaving={updatePreset.isPending}
                isDeleting={deletePreset.isPending}
                hasChanges={hasChanges}
                onSubmit={handleSave}
                onDelete={handleDelete}
                onChange={() => setHasChanges(true)}
                presetName={preset.name}
              />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Memory Tab */}
        <TabsContent value="memory">
          <AgentMemoryTab agentId={presetId} />
        </TabsContent>
      </Tabs>

      {/* Clear Conversation AlertDialog */}
      <AlertDialog open={showClearConversationDialog} onOpenChange={setShowClearConversationDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t('detail.clearConversationTitle')}</AlertDialogTitle>
            <AlertDialogDescription>{t('detail.clearConversationConfirm')}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{tc('actions.cancel')}</AlertDialogCancel>
            <AlertDialogAction onClick={handleConfirmClearAndApply}>{tc('actions.confirm')}</AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Publish Dialog */}
      <PublishDialog
        open={showPublishDialog}
        onOpenChange={setShowPublishDialog}
        modelProviders={modelProviders}
        initialModelProvider={preset.model_provider || null}
        initialModelName={preset.model_name || null}
        onPublish={handlePublish}
        isPublishing={isPublishing}
      />
    </div>
  );
}
