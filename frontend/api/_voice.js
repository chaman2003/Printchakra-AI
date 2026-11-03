// Shared voice assistant definition. The leading underscore keeps Vercel from exposing this file as a route.

const LIVE_SYSTEM_INSTRUCTION = `You are the PrintChakra voice assistant. PrintChakra is a document scanning, OCR and printing web app.
Keep every reply to one or two short spoken sentences. Speak naturally in English.

You control the app with tools:
- navigate: switch pages (dashboard, playground, phone capture, home).
- app_command: run a Dashboard action such as starting a print or scan, selecting documents, changing print settings, or running OCR.

Rules:
- When the user asks for an action, call the matching tool first, then say briefly what you did.
- "Print" or "scan" starts that workflow: call app_command with start_print or start_scan. Then use select_document, set_* commands, and proceed_action to move through the steps. Use confirm only when the user clearly says to start printing or scanning.
- To read text from a document, call app_command with run_ocr.
- Document numbers are 1-based, as the user sees them in the list.
- If a tool result says it failed, tell the user plainly. Never claim an action succeeded without a successful tool result.
- For general questions, answer directly without tools.`;

const settingProps = {
  document_number: { type: 'integer', description: '1-based document position in the list' },
  document_numbers: { type: 'array', items: { type: 'integer' }, description: 'Several 1-based document positions' },
  count: { type: 'integer', description: 'How many documents, copies or feed pages' },
  selection_type: { type: 'string', enum: ['first_n', 'last_n', 'all', 'specific'] },
  section: { type: 'string', enum: ['current', 'converted', 'upload'], description: 'Which document list' },
  filename: { type: 'string', description: 'Exact file name, only if the user gave one' },
  layout: { type: 'string', enum: ['portrait', 'landscape'] },
  colorMode: { type: 'string', enum: ['color', 'bw'] },
  paperSize: { type: 'string', description: 'A4, A3, Letter, Legal ...' },
  resolution: { type: 'integer', description: 'DPI, for example 300' },
  copies: { type: 'integer' },
  duplex: { type: 'boolean', description: 'Double-sided printing' },
  quality: { type: 'string', enum: ['draft', 'normal', 'high'] },
  margins: { type: 'string', enum: ['default', 'none', 'minimal', 'custom'] },
  scale: { type: 'integer', description: 'Percent' },
  pages: { type: 'string', enum: ['all', 'odd', 'even', 'custom'] },
  customRange: { type: 'string', description: 'Page range such as 1-3,5' },
  format: { type: 'string', enum: ['pdf', 'jpg', 'png'] },
  enabled: { type: 'boolean' },
};

const APP_COMMANDS = [
  'start_print', 'start_scan', 'run_ocr',
  'select_document', 'select_multiple_documents', 'select_documents', 'deselect_document',
  'switch_section', 'next_document', 'previous_document', 'upload_document',
  'set_layout', 'set_color', 'set_paper_size', 'set_resolution', 'set_copies', 'set_duplex',
  'set_quality', 'set_margins', 'set_scale', 'set_pages', 'set_format', 'toggle_ocr',
  'proceed_action', 'go_back', 'confirm', 'cancel', 'status', 'repeat_settings',
  'show_converted', 'device_info', 'check_connectivity', 'close_panel',
];

const LIVE_TOOLS = [
  {
    functionDeclarations: [
      {
        name: 'navigate',
        description: 'Open a page of the app.',
        parametersJsonSchema: {
          type: 'object',
          properties: { page: { type: 'string', enum: ['dashboard', 'playground', 'phone', 'home'] } },
          required: ['page'],
        },
      },
      {
        name: 'app_command',
        description: 'Run an action on the Dashboard. The Dashboard page is opened automatically if needed.',
        parametersJsonSchema: {
          type: 'object',
          properties: {
            command: { type: 'string', enum: APP_COMMANDS },
            params: { type: 'object', properties: settingProps, description: 'Only the fields the command needs' },
          },
          required: ['command'],
        },
      },
    ],
  },
];

module.exports = { LIVE_SYSTEM_INSTRUCTION, LIVE_TOOLS };
