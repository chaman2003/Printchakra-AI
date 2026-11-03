/**
 * DocumentSelector Component
 * Modern tabbed modal for selecting documents from multiple sources
 * Features: Current Documents, Converted Documents, Local Upload
 */

import React, {
  useState,
  useCallback,
  useRef,
  forwardRef,
  useImperativeHandle,
  useEffect,
} from 'react';
import {
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalBody,
  ModalCloseButton,
  ModalFooter,
  Button,
  HStack,
  Text,
  Badge,
  useColorModeValue,
} from '@chakra-ui/react';
import Iconify from '../common/Iconify';
import { Document } from '../../types';
import { processFileForPreview } from '../../utils/pdfUtils';
import DocumentSelectorTabs from './DocumentSelectorTabs';

export interface DocumentSelectorProps {
  isOpen: boolean;
  onClose: () => void;
  onSelect: (documents: Document[]) => void;
  currentDocuments?: Document[];
  convertedDocuments?: Document[];
  allowMultiple?: boolean;
  mode: 'scan' | 'print';
  isChatVisible?: boolean;
  chatWidth?: number;
  // Optional: a list of selected filenames from parent to keep selection in sync
  selectedFilenames?: string[];
}

export interface DocumentSelectorHandle {
  focusSection: (section: 'current' | 'converted' | 'upload') => void;
  selectDocumentByIndex: (section: 'current' | 'converted' | 'upload', position: number) => Document | null;
  selectMultipleDocuments: (section: 'current' | 'converted' | 'upload', indices: number[]) => void;
  addDocumentsToSelection: (section: 'current' | 'converted' | 'upload', indices: number[]) => void;
  removeDocumentsFromSelection: (section: 'current' | 'converted' | 'upload', indices: number[]) => void;
  undoLastSelection: () => void;
  clearSelections: () => void;
  getSelectionCount: () => number;
  getActiveSection: () => 'current' | 'converted' | 'upload';
  getSelectedDocuments: () => Document[];
  selectAll: () => void;
  deselectAll: () => void;
}



const TAB_INDEX: Record<'current' | 'converted' | 'upload', number> = {
  current: 0,
  converted: 1,
  upload: 2,
};

const DocumentSelector = forwardRef<DocumentSelectorHandle, DocumentSelectorProps>(
  ({
    isOpen,
    onClose,
    onSelect,
    currentDocuments = [],
    convertedDocuments = [],
    allowMultiple = true,
    mode,
    isChatVisible = false,
    chatWidth = 380,
    selectedFilenames = [],
  }, ref) => {
    const [selectedDocs, setSelectedDocs] = useState<Set<string>>(new Set());
    const [lastClickedIndex, setLastClickedIndex] = useState<number | null>(null);
    const [uploadedFiles, setUploadedFiles] = useState<Document[]>([]);
    const [isDragging, setIsDragging] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const [activeTab, setActiveTab] = useState<number>(0);

    // Sync incoming selected filenames from parent (e.g., voice selection) - ADDITIVE
    useEffect(() => {
      if (selectedFilenames && selectedFilenames.length > 0) {
        // ADDITIVE: Merge with existing selection instead of replacing
        setSelectedDocs(prev => {
          const newSet = new Set(prev);
          selectedFilenames.forEach(fn => newSet.add(fn));
          return newSet;
        });
        // try to set lastClickedIndex to first matched index
        const docs = [...currentDocuments, ...convertedDocuments];
        const first = selectedFilenames[0];
        const idx = docs.findIndex(d => d.filename === first);
        if (idx >= 0) setLastClickedIndex(idx);
      }
    }, [selectedFilenames, currentDocuments, convertedDocuments]);


    // History stack for undo/revert
    const [historyStack, setHistoryStack] = useState<Set<string>[]>([]);

    // Function to save current state to history
    const saveToHistory = useCallback(() => {
      setHistoryStack(prev => [...prev, new Set(selectedDocs)]);
    }, [selectedDocs]);

    // Updated to match SurfaceCard theme - fully opaque for better visibility
    const bgColor = useColorModeValue('white', 'gray.900');
    const hoverBg = useColorModeValue('orange.50', 'whiteAlpha.100');
    const selectedBg = useColorModeValue('brand.50', 'rgba(17, 17, 17,0.15)');
    const borderColor = useColorModeValue('gray.200', 'gray.700');

    const handleDocumentClick = useCallback(
      (filename: string, index: number, shiftKey: boolean, docs: Document[]) => {
        saveToHistory();

        if (!allowMultiple) {

          setSelectedDocs(new Set([filename]));
          setLastClickedIndex(index);
          return;
        }

        if (shiftKey && lastClickedIndex !== null) {
          // Range select
          const start = Math.min(lastClickedIndex, index);
          const end = Math.max(lastClickedIndex, index);
          const newSelected = new Set(selectedDocs);

          for (let i = start; i <= end; i++) {
            if (docs[i]) {
              newSelected.add(docs[i].filename);
            }
          }

          setSelectedDocs(newSelected);
        } else {
          // Single toggle
          const newSelected = new Set(selectedDocs);
          if (newSelected.has(filename)) {
            newSelected.delete(filename);
          } else {
            newSelected.add(filename);
          }
          setSelectedDocs(newSelected);
          setLastClickedIndex(index);
        }
      },
      [allowMultiple, lastClickedIndex, selectedDocs]
    );

    const [isUploading, setIsUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState<string>('');

    const handleFileUpload = useCallback(
      async (files: FileList | null) => {
        if (!files || files.length === 0) {
          return;
        }

        const validExtensions =
          mode === 'print' ? ['pdf', 'jpg', 'jpeg', 'png'] : ['jpg', 'jpeg', 'png'];

        setIsUploading(true);
        const processedDocs: Document[] = [];

        for (let i = 0; i < files.length; i++) {
          const file = files[i];
          const ext = file.name.split('.').pop()?.toLowerCase();

          if (!ext || !validExtensions.includes(ext)) {
            continue;
          }

          try {
            setUploadProgress(`Processing ${file.name}... (${i + 1}/${files.length})`);

            // Use PDF.js to convert PDF to images, or FileReader for images
            const result = await processFileForPreview(file);

            processedDocs.push({
              filename: result.filename,
              size: result.size,
              type: result.type,
              fileObject: result.fileObject,
              thumbnailUrl: result.thumbnailUrl,
              pages: result.pages,
            });

          } catch (error) {
            console.error(`[DocumentSelector] Error processing ${file.name}:`, error);
          }
        }

        setUploadedFiles(prev => [...prev, ...processedDocs]);
        setIsUploading(false);
        setUploadProgress('');
      },
      [mode]
    );

    const handleConfirm = () => {
      console.log('[DOC_SELECTOR] handleConfirm called');
      console.log('[DOC_SELECTOR] selectedDocs Set:', selectedDocs);
      console.log('[DOC_SELECTOR] currentDocuments:', currentDocuments);
      console.log('[DOC_SELECTOR] convertedDocuments:', convertedDocuments);
      console.log('[DOC_SELECTOR] uploadedFiles:', uploadedFiles);

      const allDocs = [...currentDocuments, ...convertedDocuments, ...uploadedFiles];
      const selected = allDocs.filter(doc => selectedDocs.has(doc.filename));

      console.log('[DOC_SELECTOR] allDocs:', allDocs);
      console.log('[DOC_SELECTOR] selected:', selected);

      // Ensure all selected documents have proper pages structure
      const docsWithPages = selected.map(doc => ({
        ...doc,
        pages: doc.pages || [{
          pageNumber: 1,
          thumbnailUrl: doc.thumbnailUrl
        }]
      }));

      console.log('[DOC_SELECTOR] docsWithPages:', docsWithPages);

      onSelect(docsWithPages);
      onClose();
    };

    useImperativeHandle(
      ref,
      () => ({
        focusSection: section => {
          if (TAB_INDEX[section] !== undefined) {
            setActiveTab(TAB_INDEX[section]);
          }
        },
        selectDocumentByIndex: (section, position) => {
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          const target = docs[position - 1];
          if (!target) {
            return null;
          }
          saveToHistory();

          setActiveTab(TAB_INDEX[section]);
          // ADDITIVE: Add to existing selection instead of replacing
          setSelectedDocs(prev => new Set([...Array.from(prev), target.filename]));


          setLastClickedIndex(position - 1);
          return target;
        },
        selectMultipleDocuments: (section, indices) => {
          saveToHistory();
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          let lastValidIndex: number | null = null;

          // ADDITIVE: Start with existing selection
          setSelectedDocs(prev => {
            const newSelected = new Set(prev);
            for (const idx of indices) {
              if (docs[idx]) {
                newSelected.add(docs[idx].filename);
                lastValidIndex = idx;
              }
            }
            return newSelected;
          });

          setActiveTab(TAB_INDEX[section]);
          if (lastValidIndex !== null) {
            setLastClickedIndex(lastValidIndex);
          }
        },
        // NEW: Add specific documents to selection (additive)
        addDocumentsToSelection: (section: 'current' | 'converted' | 'upload', indices: number[]) => {
          saveToHistory();
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          setSelectedDocs(prev => {
            const newSelected = new Set(prev);
            for (const idx of indices) {
              if (docs[idx]) {
                newSelected.add(docs[idx].filename);
              }
            }
            return newSelected;
          });
          setActiveTab(TAB_INDEX[section]);
        },
        // NEW: Remove specific documents from selection
        removeDocumentsFromSelection: (section: 'current' | 'converted' | 'upload', indices: number[]) => {
          saveToHistory();
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          setSelectedDocs(prev => {
            const newSelected = new Set(prev);
            for (const idx of indices) {
              if (docs[idx]) {
                newSelected.delete(docs[idx].filename);
              }
            }
            return newSelected;
          });
        },
        clearSelections: () => {
          saveToHistory();
          setSelectedDocs(new Set());
          setLastClickedIndex(null);
        },
        undoLastSelection: () => {
          if (historyStack.length === 0) return;
          const newHistory = [...historyStack];
          const lastState = newHistory.pop();
          setHistoryStack(newHistory);
          if (lastState) {
            setSelectedDocs(lastState);
          }
        },
        // NEW: Get current selection count
        getSelectionCount: () => selectedDocs.size,
        // NEW: Get currently active section
        getActiveSection: () => {
          const sections: ('current' | 'converted' | 'upload')[] = ['current', 'converted', 'upload'];
          return sections[activeTab] || 'current';
        },
        // NEW: Get all currently selected documents
        getSelectedDocuments: () => {
          const allDocs = [...currentDocuments, ...convertedDocuments, ...uploadedFiles];
          return allDocs.filter(doc => selectedDocs.has(doc.filename));
        },
        selectAll: () => {
          saveToHistory();
          const section = TAB_INDEX_TO_SECTION[activeTab];
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          setSelectedDocs(prev => {
            const newSet = new Set(prev);
            docs.forEach(d => newSet.add(d.filename));
            return newSet;
          });
        },
        deselectAll: () => {
          saveToHistory();
          const section = TAB_INDEX_TO_SECTION[activeTab];
          const docs = section === 'current' ? currentDocuments : section === 'converted' ? convertedDocuments : uploadedFiles;
          setSelectedDocs(prev => {
            const newSet = new Set(prev);
            docs.forEach(d => newSet.delete(d.filename));
            return newSet;
          });
        },
      }),

      [currentDocuments, convertedDocuments, uploadedFiles, selectedDocs, historyStack, saveToHistory, activeTab]
    );

    const TAB_INDEX_TO_SECTION: Record<number, 'current' | 'converted' | 'upload'> = {
      0: 'current',
      1: 'converted',
      2: 'upload',
    };




    return (
      <Modal
        isOpen={isOpen}
        onClose={onClose}
        size={isChatVisible ? "4xl" : "6xl"}
        scrollBehavior="inside"
        isCentered={!isChatVisible}
        closeOnEsc={true}
        closeOnOverlayClick={!isChatVisible}
        blockScrollOnMount={false}
        trapFocus={!isChatVisible}
        autoFocus={!isChatVisible}
        returnFocusOnClose={!isChatVisible}
      >
        <ModalOverlay
          bg="blackAlpha.600"
          backdropFilter="none"
          pointerEvents="auto"
        />
        <ModalContent
          bg={bgColor}
          maxH="90vh"
          borderRadius="2xl"
          boxShadow="0 30px 80px rgba(0, 0, 0, 0.5)"
          pointerEvents="auto"
          border="2px solid"
          borderColor="brand.500"
          maxW={isChatVisible ? `calc(100vw - ${chatWidth + 20}px)` : "1200px"}
          w={isChatVisible ? `calc(100vw - ${chatWidth + 20}px)` : "95vw"}
          ml={isChatVisible ? "8px" : "auto"}
          mr={isChatVisible ? `${chatWidth + 12}px` : "auto"}
          mt={isChatVisible ? "8px" : "auto"}
        >
          <ModalHeader
            fontSize="2xl"
            fontWeight="700"
            borderBottom="1px solid"
            borderColor={borderColor}
          >
            <HStack spacing={3}>
              <Iconify icon="solar:document-add-bold-duotone" width={28} height={28} />
              <Text>Select Documents</Text>
              {selectedDocs.size > 0 && (
                <Badge colorScheme="brand" fontSize="md" px={3} py={1}>
                  {selectedDocs.size} selected
                </Badge>
              )}
            </HStack>
          </ModalHeader>
          <ModalCloseButton size="lg" />

          <ModalBody py={6}>
            <DocumentSelectorTabs
              activeTab={activeTab}
              setActiveTab={setActiveTab}
              currentDocuments={currentDocuments}
              convertedDocuments={convertedDocuments}
              uploadedFiles={uploadedFiles}
              selectedDocs={selectedDocs}
              borderColor={borderColor}
              selectedBg={selectedBg}
              isDragging={isDragging}
              setIsDragging={setIsDragging}
              fileInputRef={fileInputRef}
              allowMultiple={allowMultiple}
              mode={mode}
              isUploading={isUploading}
              uploadProgress={uploadProgress}
              handleFileUpload={handleFileUpload}
              handleDocumentClick={handleDocumentClick}
              onSelectAll={() => ref && 'current' in ref && (ref as any).current?.selectAll()}
              onDeselectAll={() => ref && 'current' in ref && (ref as any).current?.deselectAll()}
            />
          </ModalBody>

          <ModalFooter borderTop="1px solid" borderColor={borderColor}>
            <HStack spacing={3}>
              <Text fontSize="sm" color="text.muted">
                {allowMultiple && 'Shift+Click for range selection'}
              </Text>
              <Button variant="ghost" onClick={onClose}>
                Cancel
              </Button>
              <Button
                colorScheme="brand"
                onClick={handleConfirm}
                isDisabled={selectedDocs.size === 0}
                leftIcon={<Iconify icon="solar:check-circle-bold" width={20} height={20} />}
              >
                Select {selectedDocs.size > 0 && `(${selectedDocs.size})`}
              </Button>
            </HStack>
          </ModalFooter>
        </ModalContent>
      </Modal>
    );
  });

DocumentSelector.displayName = 'DocumentSelector';

export default DocumentSelector;
